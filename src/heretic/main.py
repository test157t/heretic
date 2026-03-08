# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
import os
import re
import base64
import io
import shutil
import sys
import tempfile
import time
import tomllib
import warnings
from dataclasses import asdict
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path
from typing import cast

import huggingface_hub
import optuna
import torch
import torch.nn.functional as F
from torch import Tensor
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from pydantic import ValidationError
from questionary import Choice
from rich.traceback import install

from .config import QuantizationMethod, Settings
from .unsloth_stage import (
    UnslothStageSettings,
    export_unsloth_checkpoint_snapshot,
    export_unsloth_checkpoint_snapshot_dual,
    quantize_snapshot_to_4bit,
    run_unsloth_stage,
)
from .utils import (
    empty_cache,
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
    print_memory_usage,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)


def resolve_config_file(primary: str, fallback: str | None = None) -> str:
    if Path(primary).exists():
        return primary

    if fallback is not None and Path(fallback).exists():
        return fallback

    raise FileNotFoundError(primary)


def load_toml_file(config_file: str) -> dict:
    with open(config_file, "rb") as file:
        return tomllib.load(file)


def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_settings_from_file(
    config_file: str,
    model_override: str | None,
    base_config_file: str | None = None,
) -> Settings:
    data = load_toml_file(config_file)

    if base_config_file is not None:
        data = merge_dicts(load_toml_file(base_config_file), data)

    if model_override:
        data["model"] = model_override

    return Settings.model_validate(data)


def load_unsloth_settings_from_file(
    config_file: str,
    model_override: str | None,
) -> UnslothStageSettings:
    data = load_toml_file(config_file)

    if model_override and "model" not in data:
        data["model"] = model_override

    return UnslothStageSettings.model_validate(data)


def serialize_tensor(tensor: Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def deserialize_tensor(data: str) -> Tensor:
    buffer = io.BytesIO(base64.b64decode(data.encode("ascii")))
    return cast(Tensor, torch.load(buffer, map_location="cpu"))


def find_latest_training_checkpoint(checkpoint_dir: str) -> str | None:
    root = Path(checkpoint_dir)
    if not root.exists():
        return None

    checkpoints = []
    for entry in root.glob("checkpoint-*"):
        if not entry.is_dir():
            continue
        try:
            step = int(entry.name.split("-")[-1])
        except Exception:
            continue
        checkpoints.append((step, entry))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def configure_hf_token_for_session():
    existing_token = huggingface_hub.get_token()

    choices = [
        Choice(
            title=(
                "Use existing Hugging Face token"
                if existing_token
                else "Continue without Hugging Face token"
            ),
            value="existing",
        ),
        Choice(
            title="Enter Hugging Face token for this session",
            value="enter",
        ),
    ]

    print()
    token_choice = prompt_select("Hugging Face authentication", choices)

    if token_choice == "enter":
        token = prompt_password("Hugging Face access token:")
        if token:
            os.environ["HF_TOKEN"] = token
            print("* Session token set")
        else:
            print("* No token entered")
    elif existing_token:
        print("* Using existing Hugging Face token")
    else:
        print("* Continuing without Hugging Face token")


def obtain_merge_strategy(settings: Settings) -> str | None:
    """
    Prompts the user for how to proceed with saving the model.
    Provides info to the user if the model is quantized on memory use.
    Returns "merge", "adapter", or None (if cancelled/invalid).
    """

    if settings.quantization == QuantizationMethod.BNB_4BIT:
        from .model import get_model_class

        print()
        print(
            "Model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{footprint_gb:.2f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or bitsandbytes quantization config issues on the meta device).
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )
        print()

        strategy = prompt_select(
            "How do you want to proceed?",
            choices=[
                Choice(
                    title="Merge LoRA into full model"
                    + (
                        ""
                        if settings.quantization == QuantizationMethod.NONE
                        else " (requires sufficient RAM)"
                    ),
                    value="merge",
                ),
                Choice(
                    title="Cancel",
                    value="cancel",
                ),
            ],
        )

        if strategy == "cancel":
            return None

        return strategy
    else:
        return "merge"


def prompt_runtime_quantization(default: QuantizationMethod) -> QuantizationMethod:
    choice = prompt_select(
        "Quantization mode",
        [
            Choice(
                title=(
                    "Use bnb_4bit"
                    if default == QuantizationMethod.BNB_4BIT
                    else "Use no quantization"
                )
                + " (recommended)",
                value=default,
            ),
            Choice(
                title=(
                    "Use no quantization"
                    if default == QuantizationMethod.BNB_4BIT
                    else "Use bnb_4bit"
                ),
                value=(
                    QuantizationMethod.NONE
                    if default == QuantizationMethod.BNB_4BIT
                    else QuantizationMethod.BNB_4BIT
                ),
            ),
        ],
    )

    return choice if choice is not None else default


def suggest_full_precision_base(model_name: str) -> str:
    lowered = model_name.lower()
    suffixes = ["-bnb-4bit", "-4bit", "-int4", "-awq"]
    for suffix in suffixes:
        if lowered.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def suggest_4bit_export_path(path_16bit: str) -> str:
    lowered = path_16bit.lower()

    replacements = [
        ("-bf16", "-bnb-4bit"),
        ("_bf16", "_bnb-4bit"),
        ("-16bit", "-bnb-4bit"),
        ("_16bit", "_bnb-4bit"),
    ]

    for suffix, replacement in replacements:
        if lowered.endswith(suffix):
            return path_16bit[: -len(suffix)] + replacement

    return f"{path_16bit}-bnb-4bit"


def chat_with_model(model, settings: Settings):
    print()
    print("[cyan]Press Ctrl+C at any time to return to the menu.[/]")

    chat = [
        {"role": "system", "content": settings.system_prompt},
    ]

    while True:
        try:
            message = prompt_text(
                "User:",
                qmark=">",
                unsafe=True,
            )
            if not message:
                break
            chat.append({"role": "user", "content": message})

            print("[bold]Assistant:[/] ", end="")
            response = model.stream_chat_response(chat)
            chat.append(
                {"role": "assistant", "content": response}
            )
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C/Ctrl+D
            break


def run():
    # Configure CUDA allocator defaults to reduce fragmentation-induced OOM.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        if os.name == "nt":
            # Windows CUDA builds may not support expandable_segments reliably.
            allocator_conf = "max_split_size_mb:128,garbage_collection_threshold:0.8"
        else:
            allocator_conf = "expandable_segments:True,max_split_size_mb:128"

        os.environ["PYTORCH_ALLOC_CONF"] = allocator_conf
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_conf

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    model_override = None
    if len(sys.argv) > 1 and not sys.argv[-1].startswith("-"):
        # Allow `heretic <model>` to override the model field in the selected config.
        model_override = sys.argv[-1]
        # Remove the positional model argument from argv so Settings-related
        # CLI parsers in dependencies cannot treat it as an unknown flag value.
        sys.argv = sys.argv[:-1]

    configure_hf_token_for_session()

    print(
        "Select workflow. Training/ablation/slop use config files: "
        "[bold]config.train.toml[/], [bold]config.ablate.toml[/], [bold]config.slop.toml[/]."
    )
    print()
    stage = prompt_select(
        "What do you want to run?",
        [
            Choice(
                title="Train with Unsloth (config.train.toml)",
                value="pre",
            ),
            Choice(
                title="Run ablation workflow (config.ablate.toml)",
                value="ablate",
            ),
            Choice(
                title="Run slop workflow (config.slop.toml)",
                value="slop",
            ),
            Choice(
                title="Quantize model to bnb_4bit",
                value="quantize",
            ),
            Choice(
                title="Exit program",
                value="",
            ),
        ],
    )

    if stage is None or stage == "":
        return

    if stage == "quantize":
        print()
        model_name = prompt_text("Model path or Hugging Face ID to quantize:")
        if not model_name:
            return

        output_default = suggest_4bit_export_path(model_name)
        output_model = prompt_text(
            "Output folder for bnb_4bit model:",
            default=output_default,
        )
        if not output_model:
            return

        try:
            print(f"Quantizing [bold]{model_name}[/] to bnb_4bit...")
            quantize_snapshot_to_4bit(model_name=model_name, output_model=output_model)
            print(f"Quantized model saved to [bold]{output_model}[/].")
        except Exception as error:
            print(f"[red]Quantization failed: {error}[/]")
        return

    if stage == "pre":
        # For Unsloth training, import unsloth before transformers/peft.
        try:
            import unsloth  # ty:ignore[unresolved-import, unused-ignore]
        except Exception:
            pass

    # Import these lazily so pre-stage can initialize Unsloth first.
    import transformers
    from .analyzer import Analyzer
    from .evaluator import Evaluator
    from .model import AbliterationParameters, Model, get_model_class

    pre_stage_settings = None
    config_file = ""
    base_config_file = None

    try:
        if stage == "pre":
            pre_config_file = resolve_config_file("config.train.toml")
            pre_stage_settings = load_unsloth_settings_from_file(
                pre_config_file,
                model_override,
            )
            config_file = resolve_config_file("config.ablate.toml", "config.toml")
            # If config.ablate.toml exists, use it as the sole source of truth.
            # Only fall back to config.toml when config.ablate.toml is missing.
            if config_file == "config.toml":
                base_config_file = None

        elif stage == "ablate":
            config_file = resolve_config_file("config.ablate.toml", "config.toml")
            # If config.ablate.toml exists, use it as the sole source of truth.
            # Only fall back to config.toml when config.ablate.toml is missing.
            if config_file == "config.toml":
                base_config_file = None

        else:
            config_file = resolve_config_file("config.slop.toml")
            if Path("config.ablate.toml").exists():
                base_config_file = "config.ablate.toml"
            elif Path("config.toml").exists():
                base_config_file = "config.toml"

        settings = load_settings_from_file(
            config_file,
            model_override,
            base_config_file=base_config_file,
        )
    except FileNotFoundError as error:
        missing = str(error) if str(error) else config_file
        print(f"[red]Could not find [bold]{missing}[/].[/]")
        return
    except ValidationError as error:
        print(
            f"[red]Configuration [bold]{config_file}[/] contains [bold]{error.error_count()}[/] errors:[/]"
        )

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print("See [bold]config.default.toml[/] for details about configuration parameters.")
        return

    if pre_stage_settings is not None and pre_stage_settings.enabled:

        input_model = pre_stage_settings.model or settings.model
        print()
        resume_checkpoint = None
        pre_checkpoint_dir = (
            pre_stage_settings.checkpoint_output
            or f"{pre_stage_settings.output_model}_checkpoints"
        )
        latest_pre_checkpoint = find_latest_training_checkpoint(pre_checkpoint_dir)
        if latest_pre_checkpoint is not None:
            checkpoint_action = prompt_select(
                "Found existing training checkpoints. How do you want to proceed?",
                [
                    "Resume from latest checkpoint",
                    "Start a fresh training run",
                    "Exit",
                ],
            )
            if checkpoint_action == "Resume from latest checkpoint":
                resume_checkpoint = latest_pre_checkpoint
            elif checkpoint_action == "Exit" or checkpoint_action is None:
                return

        use_checkpoint_runtime = False
        runtime_checkpoint = None

        def print_export_verification(report):
            if not isinstance(report, dict):
                return

            lora_signal = report.get("lora_signal")
            merge_deltas = report.get("merge_deltas")
            if not isinstance(lora_signal, dict) or not isinstance(merge_deltas, dict):
                return

            lora_b_nonzero = int(lora_signal.get("lora_b_nonzero_tensors", 0))
            lora_b_total = int(lora_signal.get("lora_b_tensors", 0))
            lora_b_abs_sum = float(lora_signal.get("lora_b_abs_sum", 0.0))
            changed_count = int(merge_deltas.get("changed_count", 0))
            compared_count = int(merge_deltas.get("compared_count", 0))
            max_delta = float(merge_deltas.get("max_mean_abs_delta", 0.0))

            print("* Merge verification")
            print(
                f"  * LoRA B tensors with signal: [bold]{lora_b_nonzero}[/]/[bold]{lora_b_total}[/] (abs_sum={lora_b_abs_sum:.4e})"
            )
            print(
                f"  * Sampled merged weights changed: [bold]{changed_count}[/]/[bold]{compared_count}[/] (max mean abs delta={max_delta:.4e})"
            )

            if lora_b_nonzero == 0 or lora_b_abs_sum <= 0.0:
                print(
                    "[yellow]  * Warning: adapter appears near-zero; training may not have updated LoRA weights.[/]"
                )
            elif changed_count == 0:
                print(
                    "[yellow]  * Warning: sampled merged weights did not change; verify checkpoint/base pairing.[/]"
                )
            else:
                print("[green]  * Adapter signal and merge deltas look non-zero.[/]")

        def save_pretrained_output(save_directory: str):
            if use_checkpoint_runtime and runtime_checkpoint is not None:
                save_mode = prompt_select(
                    "How do you want to save the pre-trained output?",
                    [
                        "Merged full model (16-bit)",
                        "Merged full model (4-bit)",
                        "Merged full models (16-bit + 4-bit)",
                        "Adapter checkpoint only",
                    ],
                )

                suggested_base = suggest_full_precision_base(input_model)

                if save_mode == "Adapter checkpoint only":
                    shutil.copytree(
                        runtime_checkpoint,
                        save_directory,
                        dirs_exist_ok=True,
                    )
                    print(f"Adapter checkpoint saved to [bold]{save_directory}[/].")
                    return

                if save_mode == "Merged full models (16-bit + 4-bit)":
                    merge_base_model = prompt_text(
                        "Full-precision base model/path for merged exports (required for pre-quantized bases):",
                        default=suggested_base,
                    )
                    if merge_base_model is not None:
                        merge_base_model = merge_base_model.strip()
                    if not merge_base_model:
                        print("[yellow]Combined export cancelled.[/]")
                        return

                    save_directory_4 = suggest_4bit_export_path(save_directory)
                    print(
                        f"Exporting merged 16-bit and 4-bit models from checkpoint [bold]{runtime_checkpoint}[/]..."
                    )
                    print(f"* 16-bit output: [bold]{save_directory}[/]")
                    print(f"* 4-bit output: [bold]{save_directory_4}[/]")
                    verification = export_unsloth_checkpoint_snapshot_dual(
                        base_model=input_model,
                        adapter_checkpoint=runtime_checkpoint,
                        output_model_16bit=save_directory,
                        output_model_4bit=save_directory_4,
                        merge_base_model=merge_base_model,
                    )
                    print(
                        f"Merged models saved to [bold]{save_directory}[/] and [bold]{save_directory_4}[/]."
                    )
                    print_export_verification(verification)
                    return

                save_method = (
                    "merged_16bit"
                    if save_mode == "Merged full model (16-bit)"
                    else "merged_4bit"
                )
                merge_base_model = None
                if save_method == "merged_16bit":
                    merge_base_model = prompt_text(
                        "Full-precision base model/path for 16-bit merge (required for pre-quantized bases):",
                        default=suggested_base,
                    )
                    if not merge_base_model:
                        print("[yellow]16-bit merge cancelled.[/]")
                        return
                else:
                    merge_base_model = prompt_text(
                        "Optional full-precision base model/path for 4-bit export (recommended for pre-quantized bases):",
                        default=suggested_base,
                    )
                    if merge_base_model == "":
                        merge_base_model = None

                print(
                    f"Exporting merged model from checkpoint [bold]{runtime_checkpoint}[/]..."
                )
                verification = export_unsloth_checkpoint_snapshot(
                    base_model=input_model,
                    adapter_checkpoint=runtime_checkpoint,
                    output_model=save_directory,
                    save_method=save_method,
                    merge_base_model=merge_base_model,
                )
                print(f"Merged model saved to [bold]{save_directory}[/].")
                print_export_verification(verification)
                return

            shutil.copytree(output_model, save_directory, dirs_exist_ok=True)
            print(f"Model saved to [bold]{save_directory}[/].")

        while True:
            print(f"Running Unsloth training using [bold]{input_model}[/]...")
            try:
                pre_result = run_unsloth_stage(
                    pre_stage_settings,
                    input_model,
                    resume_from_checkpoint=resume_checkpoint,
                    export_snapshot=False,
                )
            except Exception as error:
                print(f"[red]Training failed: {error}[/]")
                return

            output_model = pre_result.output_model
            latest_checkpoint = pre_result.latest_checkpoint

            if (
                latest_checkpoint is not None
                and Path(output_model).name.startswith("checkpoint-")
            ):
                use_checkpoint_runtime = True
                runtime_checkpoint = latest_checkpoint

            if pre_result.interrupted:
                print(
                    f"* Training interrupted. Snapshot available at [bold]{output_model}[/]."
                )
                action = prompt_select(
                    "Training interrupted. What now?",
                    [
                        "Resume training from latest checkpoint",
                        "Continue using latest checkpoint adapter",
                        "Chat with current snapshot",
                        "Save current snapshot to a local folder",
                        "Exit (I will run later)",
                    ],
                )

                if action == "Resume training from latest checkpoint":
                    if latest_checkpoint is None:
                        print("[yellow]No checkpoint found to resume from.[/]")
                        resume_checkpoint = None
                    else:
                        resume_checkpoint = latest_checkpoint
                    print()
                    continue
                if action == "Chat with current snapshot":
                    print()
                    print("Loading pre-trained model for chat...")
                    chat_settings = settings.model_copy(deep=True)
                    if latest_checkpoint is not None:
                        chat_settings.model = input_model
                        chat_settings.initial_adapter_path = latest_checkpoint
                        print(
                            f"* Using base model [bold]{input_model}[/] with adapter checkpoint [bold]{latest_checkpoint}[/]"
                        )
                    else:
                        chat_settings.model = output_model
                    chat_settings.quantization = prompt_runtime_quantization(
                        chat_settings.quantization
                    )
                    chat_model = Model(chat_settings)
                    print()
                    print_memory_usage()
                    chat_with_model(chat_model, chat_settings)
                    print()
                    continue
                if action == "Save current snapshot to a local folder":
                    save_directory = prompt_path("Path to the folder:")
                    if save_directory:
                        save_pretrained_output(save_directory)
                    print()
                    continue
                if action is None or action == "Exit (I will run later)":
                    return
                if action == "Continue using latest checkpoint adapter":
                    if latest_checkpoint is None:
                        print(
                            "[yellow]No checkpoint found. Continuing with snapshot directory instead.[/]"
                        )
                    else:
                        use_checkpoint_runtime = True
                        runtime_checkpoint = latest_checkpoint
            else:
                print(f"* Training completed. Output saved to [bold]{output_model}[/].")
            break

        while True:
            pre_action = prompt_select(
                "What do you want to do next?",
                [
                    "Save trained model to a local folder",
                    "Chat with trained model",
                    "Exit",
                ],
            )
            if pre_action is None or pre_action == "Exit":
                return

            if pre_action == "Save trained model to a local folder":
                save_directory = prompt_path("Path to the folder:")
                if not save_directory:
                    continue

                save_pretrained_output(save_directory)
                continue

            print()
            print(f"Loading trained model [bold]{output_model}[/] for chat...")
            chat_settings = settings.model_copy(deep=True)
            if use_checkpoint_runtime and runtime_checkpoint is not None:
                chat_settings.model = input_model
                chat_settings.initial_adapter_path = runtime_checkpoint
            else:
                chat_settings.model = output_model
            if "float32" in chat_settings.dtypes and chat_settings.dtypes[0] != "float32":
                chat_settings.dtypes = ["float32"] + [
                    dtype for dtype in chat_settings.dtypes if dtype != "float32"
                ]

            chat_model = Model(chat_settings)
            print()
            print_memory_usage()
            chat_with_model(chat_model, chat_settings)

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        total_vram = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))
        print(
            f"Detected [bold]{count}[/] CUDA device(s) ({total_vram / (1024**3):.2f} GB total VRAM):"
        )
        for i in range(count):
            vram = torch.cuda.mem_get_info(i)[1] / (1024**3)
            print(
                f"* GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/] ({vram:.2f} GB)"
            )
    elif is_xpu_available():
        count = torch.xpu.device_count()
        print(f"Detected [bold]{count}[/] XPU device(s):")
        for i in range(count):
            print(f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]")
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MLU device(s):")
        for i in range(count):
            print(f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] SDAA device(s):")
        for i in range(count):
            print(f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MUSA device(s):")
        for i in range(count):
            print(f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        print(f"NPU detected (CANN version: [bold]{torch.version.cann}[/])")  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        print("Detected [bold]1[/] MPS device (Apple Metal)")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)

    study_checkpoint_file = os.path.join(
        settings.study_checkpoint_dir,
        "".join(
            [(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model]
        )
        + ".jsonl",
    )

    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        existing_study = None

    if existing_study is not None and settings.evaluate_model is None:
        choices = []

        if existing_study.user_attrs["finished"]:
            print()
            print(
                (
                    "[green]You have already processed this model.[/] "
                    "You can load the previous run (this reloads the model and recomputes refusal directions) to export/chat or run additional trials. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Load previous run results",
                    value="continue",
                )
            )
        else:
            print()
            print(
                (
                    "[yellow]You have already processed this model, but the run was interrupted.[/] "
                    "You can continue the previous run from where it stopped. This will override any specified settings. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Continue the previous run",
                    value="continue",
                )
            )

        choices.append(
            Choice(
                title="Ignore the previous run and start from scratch",
                value="restart",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        choice = prompt_select("How would you like to proceed?", choices)

        if choice == "continue":
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
            print("* Loading previous run state...")
        elif choice == "restart":
            os.unlink(study_checkpoint_file)
            backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
            storage = JournalStorage(backend)
        elif choice is None or choice == "":
            return

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name="heretic",
        load_if_exists=True,
    )

    study.set_user_attr("settings", settings.model_dump_json())
    if "finished" not in study.user_attrs:
        study.set_user_attr("finished", False)

    model = Model(settings)
    print()
    print_memory_usage()

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    print()
    print("Checking response prefix...")

    prefix_detection_enabled = (
        settings.detect_reasoning_block_prefix
        or settings.detect_common_response_prefix
    )

    if not prefix_detection_enabled:
        model.response_prefix = ""
        print("* Response prefix detection disabled")
    else:
        prefix_check_prompts = good_prompts[:100] + bad_prompts[:100]
        responses = model.get_responses_batched(prefix_check_prompts)

    def canonicalize_reasoning_prefix(prefix: str) -> str | None:
        def strip_control_prefix(text: str) -> str:
            # Drop leading control tokens and whitespace (e.g. <|...|>, <s>, [INST])
            # so we can robustly detect reasoning wrappers that start a little later.
            # Intentionally do not remove generic <...> / [...] tags, because that
            # would also strip reasoning wrappers like <think> or [THINK].
            while True:
                updated = re.sub(
                    r"^(?:\s+|<\|[^|]+\|>|</?s>|\[/?INST\]|\[/?SYSTEM\]|\[/?USER\]|\[/?ASSISTANT\])",
                    "",
                    text,
                )
                if updated == text:
                    return text
                text = updated

        normalized = strip_control_prefix(prefix)
        normalized_lower = normalized.lower()

        if normalized_lower.startswith("<think>"):
            # Most thinking models.
            return "<think></think>"
        if normalized.startswith("<|channel|>analysis<|message|>"):
            # gpt-oss.
            return "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
        if normalized_lower.startswith("<thought>"):
            # Unknown, suggested by user.
            return "<thought></thought>"
        if normalized_lower.startswith("[think]"):
            # Unknown, suggested by user.
            return "[THINK][/THINK]"

        return None

    def detect_reasoning_prefix(responses: list[str]) -> tuple[str, int]:
        prefix_counts: dict[str, int] = {}

        for response in responses:
            response = response.lstrip()
            prefix = canonicalize_reasoning_prefix(response)
            if prefix is None:
                continue
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        if not prefix_counts:
            return "", 0

        return max(prefix_counts.items(), key=lambda item: item[1])

    if prefix_detection_enabled:
        recheck_prefix = False
        model.response_prefix = ""

        if settings.detect_reasoning_block_prefix:
            detected_prefix, detected_count = detect_reasoning_prefix(responses)
            if detected_prefix:
                model.response_prefix = detected_prefix
                recheck_prefix = True
                print(
                    f"* Reasoning block detected in [bold]{detected_count}[/]/{len(prefix_check_prompts)} responses"
                )
            else:
                print("* No reasoning block prefix detected")

        if not model.response_prefix and settings.detect_common_response_prefix:
            # Despite being located in os.path, commonprefix actually performs
            # a naive string operation without any path-specific logic,
            # which is exactly what we need here. Trailing spaces are removed
            # to avoid issues where multiple different tokens that all start
            # with a space character lead to the common prefix ending with
            # a space, which would result in an uncommon tokenization.
            model.response_prefix = commonprefix(responses).rstrip(" ")

            canonical_prefix = canonicalize_reasoning_prefix(model.response_prefix)
            if canonical_prefix is not None:
                # When using predefined prefixes, we need to check that the
                # prefix is actually complete (e.g. not missing trailing newlines).
                model.response_prefix = canonical_prefix
                recheck_prefix = True
        elif not model.response_prefix:
            print("* Common prefix fallback disabled")

        if model.response_prefix:
            print(f"* Prefix found: [bold]{model.response_prefix!r}[/]")
        else:
            print("* None found")

        if recheck_prefix:
            print("* Rechecking with prefix...")
            responses = model.get_responses_batched(prefix_check_prompts)
            additional_prefix = commonprefix(responses).rstrip(" ")
            if additional_prefix:
                model.response_prefix += additional_prefix
                print(f"* Extended prefix found: [bold]{model.response_prefix!r}[/]")

    evaluator = None

    def get_evaluator() -> Evaluator:
        nonlocal evaluator
        if evaluator is None:
            evaluator = Evaluator(settings, model)
        return evaluator

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        get_evaluator().get_score()
        return

    print()
    print("Preparing refusal directions...")
    refusal_directions = None
    serialized_refusal_directions = study.user_attrs.get("refusal_directions")
    if isinstance(serialized_refusal_directions, str):
        try:
            refusal_directions = deserialize_tensor(serialized_refusal_directions)
            print("* Loaded saved refusal directions from previous run")
        except Exception:
            refusal_directions = None

    def get_refusal_directions() -> Tensor:
        nonlocal refusal_directions

        if refusal_directions is not None:
            return refusal_directions

        print("* Calculating per-layer refusal directions...")
        print("* Obtaining residuals for good prompts...")
        good_residuals = model.get_residuals_batched(good_prompts)
        print("* Obtaining residuals for bad prompts...")
        bad_residuals = model.get_residuals_batched(bad_prompts)

        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)

        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

        if settings.orthogonalize_direction:
            # Implements https://huggingface.co/blog/grimjim/projected-abliteration
            # Adjust the refusal directions so that only the component that is
            # orthogonal to the good direction is subtracted during abliteration.
            good_directions = F.normalize(good_means, p=2, dim=1)
            projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
            refusal_directions = (
                refusal_directions - projection_vector.unsqueeze(1) * good_directions
            )
            refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

        analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

        if settings.print_residual_geometry:
            analyzer.print_residual_geometry()

        if settings.plot_residuals:
            analyzer.plot_residuals()

        # We don't need the residuals after computing refusal directions.
        del good_residuals, bad_residuals, analyzer
        empty_cache()

        study.set_user_attr("refusal_directions", serialize_tensor(refusal_directions))
        return refusal_directions

    trial_index = 0
    start_index = 0
    start_time = time.perf_counter()

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer_index,
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        model.reset_model()
        print("* Abliterating...")
        model.abliterate(get_refusal_directions(), direction_index, parameters)
        print("* Evaluating...")
        (
            score,
            kl_divergence,
            hellinger_distance,
            top5_ordered,
            top10_unordered,
            refusals,
        ) = get_evaluator().get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / (trial_index - start_index)) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )
        print_memory_usage()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("hellinger_distance", hellinger_distance)
        trial.set_user_attr("top_5_ordered", top5_ordered)
        trial.set_user_attr("top_10_unordered", top10_unordered)
        trial.set_user_attr("refusals", refusals)

        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            # Stop the study gracefully on Ctrl+C.
            trial.study.stop()
            raise TrialPruned()

    def count_completed_trials() -> int:
        # Count number of complete trials to compute trials to run.
        return sum([(1 if t.state == TrialState.COMPLETE else 0) for t in study.trials])

    def ensure_optimization_baselines():
        # Make sure baseline metrics and refusal directions are computed from
        # the clean model state (before any trial-specific abliteration).
        print()
        print("Preparing optimization baselines...")
        print("* Resetting model...")
        model.reset_model()
        get_refusal_directions()
        print("* Resetting model...")
        model.reset_model()
        evaluator = get_evaluator()
        study.set_user_attr("bad_evaluation_prompt_count", len(evaluator.bad_prompts))

    start_index = trial_index = count_completed_trials()
    if start_index > 0:
        print()
        print("Resuming existing study.")

    try:
        initial_trials_to_run = settings.n_trials - count_completed_trials()
        if initial_trials_to_run > 0:
            ensure_optimization_baselines()

        study.optimize(
            objective_wrapper,
            n_trials=initial_trials_to_run,
        )
    except KeyboardInterrupt:
        # This additional handler takes care of the small chance that KeyboardInterrupt
        # is raised just between trials, which wouldn't be caught by the handler
        # defined in objective_wrapper above.
        pass

    if count_completed_trials() == settings.n_trials:
        study.set_user_attr("finished", True)

    while True:
        # If no trials at all have been evaluated, the study must have been stopped
        # by pressing Ctrl+C while the first trial was running. In this case, we just
        # re-raise the interrupt to invoke the standard handler defined below.
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise KeyboardInterrupt

        # Get the Pareto front of trials. We can't use study.best_trials directly
        # as get_score() doesn't return the pure KL divergence and refusal count.
        # Note: Unlike study.best_trials, this does not handle objective constraints.
        sorted_trials = sorted(
            completed_trials,
            key=lambda trial: (
                trial.user_attrs["refusals"],
                trial.user_attrs["kl_divergence"],
            ),
        )
        min_divergence = math.inf
        best_trials = []
        for trial in sorted_trials:
            kl_divergence = trial.user_attrs["kl_divergence"]
            if kl_divergence < min_divergence:
                min_divergence = kl_divergence
                best_trials.append(trial)

        bad_eval_count = cast(int, study.user_attrs.get("bad_evaluation_prompt_count", 0))
        if bad_eval_count <= 0:
            bad_eval_count = len(load_prompts(settings, settings.bad_evaluation_prompts))
            study.set_user_attr("bad_evaluation_prompt_count", bad_eval_count)

        choices = [
            Choice(
                title=(
                    f"[Trial {trial.user_attrs['index']:>3}] "
                    f"Refusals: {trial.user_attrs['refusals']:>2}/{bad_eval_count}, "
                    f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}, "
                    f"Hellinger distance: {trial.user_attrs.get('hellinger_distance', float('nan')):.4f}, "
                    f"Top 5 ordered: {trial.user_attrs.get('top_5_ordered', float('nan')):.1%}, "
                    f"Top 10 unordered: {trial.user_attrs.get('top_10_unordered', float('nan')):.1%}"
                ),
                value=trial,
            )
            for trial in best_trials
        ]

        choices.append(
            Choice(
                title="Run additional trials",
                value="continue",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        print("[bold green]Optimization finished![/]")
        print()
        print(
            (
                "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
                "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
            )
        )

        while True:
            print()
            trial = prompt_select("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_additional_trials = prompt_text(
                            "How many additional trials do you want to run?"
                        )
                        if n_additional_trials is None or n_additional_trials == "":
                            n_additional_trials = 0
                            break
                        n_additional_trials = int(n_additional_trials)
                        if n_additional_trials > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Please enter a number.[/]")

                if n_additional_trials == 0:
                    continue

                settings.n_trials += n_additional_trials
                study.set_user_attr("settings", settings.model_dump_json())
                study.set_user_attr("finished", False)

                try:
                    additional_trials_to_run = settings.n_trials - count_completed_trials()
                    if additional_trials_to_run > 0:
                        ensure_optimization_baselines()

                    study.optimize(
                        objective_wrapper,
                        n_trials=additional_trials_to_run,
                    )
                except KeyboardInterrupt:
                    pass

                if count_completed_trials() == settings.n_trials:
                    study.set_user_attr("finished", True)

                break

            elif trial is None or trial == "":
                return

            print()
            print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")
            print("* Resetting model...")
            model.reset_model()
            print("* Abliterating...")
            model.abliterate(
                get_refusal_directions(),
                trial.user_attrs["direction_index"],
                {
                    k: AbliterationParameters(**v)
                    for k, v in trial.user_attrs["parameters"].items()
                },
            )

            def save_decensored_model(save_mode: str):
                save_directory = prompt_path("Path to the folder:")
                if not save_directory:
                    return

                if save_mode == "adapter":
                    print("Saving LoRA adapter...")
                    model.model.save_pretrained(save_directory)
                    print(f"Model saved to [bold]{save_directory}[/].")
                    return

                merge_base_model = None
                if (
                    settings.quantization == QuantizationMethod.BNB_4BIT
                    and getattr(model, "prequantized_bnb4bit", False)
                ):
                    merge_base_model = prompt_text(
                        "Full-precision base model/path for 16-bit merge:",
                        default=suggest_full_precision_base(settings.model),
                    )
                    if not merge_base_model:
                        print("[yellow]Merge cancelled.[/]")
                        return

                print("Saving merged model...")
                merged_model = model.get_merged_model(
                    merge_base_model=merge_base_model,
                )

                output_dir_16 = save_directory
                output_dir_4 = suggest_4bit_export_path(save_directory)

                if save_mode == "merged_4bit":
                    temp_16_dir = tempfile.mkdtemp(prefix="heretic-merge16-")
                    try:
                        merged_model.save_pretrained(temp_16_dir)
                        model.tokenizer.save_pretrained(temp_16_dir)
                        del merged_model
                        empty_cache()
                        quantize_snapshot_to_4bit(
                            model_name=temp_16_dir,
                            output_model=save_directory,
                        )
                    finally:
                        shutil.rmtree(temp_16_dir, ignore_errors=True)
                    print(f"Model saved to [bold]{save_directory}[/].")
                    return

                merged_model.save_pretrained(output_dir_16)
                del merged_model
                empty_cache()
                model.tokenizer.save_pretrained(output_dir_16)

                if save_mode == "merged_16bit_4bit":
                    quantize_snapshot_to_4bit(
                        model_name=output_dir_16,
                        output_model=output_dir_4,
                    )
                    print(
                        f"Models saved to [bold]{output_dir_16}[/] and [bold]{output_dir_4}[/]."
                    )
                    return

                print(f"Model saved to [bold]{save_directory}[/].")

            while True:
                print()
                action = prompt_select(
                    "What do you want to do with the decensored model?",
                    [
                        "Save merged model to local folder (16-bit)",
                        "Save merged model to local folder (4-bit)",
                        "Save merged models to local folders (16-bit + 4-bit)",
                        "Save LoRA adapter to local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Return to the trial selection menu",
                    ],
                )

                if action is None or action == "Return to the trial selection menu":
                    break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "Save merged model to local folder (16-bit)":
                            save_decensored_model("merged_16bit")

                        case "Save merged model to local folder (4-bit)":
                            save_decensored_model("merged_4bit")

                        case "Save merged models to local folders (16-bit + 4-bit)":
                            save_decensored_model("merged_16bit_4bit")

                        case "Save LoRA adapter to local folder":
                            save_decensored_model("adapter")

                        case "Upload the model to Hugging Face":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                token = prompt_password("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = prompt_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(settings.model).name}-heretic",
                            )

                            visibility = prompt_select(
                                "Should the repository be public or private?",
                                [
                                    "Public",
                                    "Private",
                                ],
                            )
                            private = visibility == "Private"

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                continue

                            if strategy == "adapter":
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merge_base_model = None
                                if (
                                    settings.quantization == QuantizationMethod.BNB_4BIT
                                    and getattr(model, "prequantized_bnb4bit", False)
                                ):
                                    merge_base_model = prompt_text(
                                        "Full-precision base model/path for 16-bit merge:",
                                        default=suggest_full_precision_base(settings.model),
                                    )
                                    if not merge_base_model:
                                        print("[yellow]Upload cancelled.[/]")
                                        continue

                                merged_model = model.get_merged_model(
                                    merge_base_model=merge_base_model,
                                )
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )

                            # If the model path exists locally and includes the
                            # card, use it directly. If the model path doesn't
                            # exist locally, it can be assumed to be a model
                            # hosted on the Hugging Face Hub, in which case
                            # we can retrieve the model card.
                            model_path = Path(settings.model)
                            if model_path.exists():
                                card_path = (
                                    model_path / huggingface_hub.constants.REPOCARD_NAME
                                )
                                if card_path.exists():
                                    card = ModelCard.load(card_path)
                                else:
                                    card = None
                            else:
                                card = ModelCard.load(settings.model)
                            if card is not None:
                                evaluator_for_card = get_evaluator()
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags.append("heretic")
                                card.data.tags.append("uncensored")
                                card.data.tags.append("decensored")
                                card.data.tags.append("abliterated")
                                card.text = (
                                    get_readme_intro(
                                        settings,
                                        trial,
                                        evaluator_for_card.base_refusals,
                                        evaluator_for_card.bad_prompts,
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                        case "Chat with the model":
                            chat_with_model(model, settings)

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
