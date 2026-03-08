# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import inspect
import json
import math
import os
from pathlib import Path
import re
from dataclasses import dataclass
import tempfile
from typing import Any

import torch
from pydantic import BaseModel, Field


class UnslothStageSettings(BaseModel):
    enabled: bool = Field(default=True)

    model: str | None = Field(default=None)
    output_model: str = Field(default="outputs/unsloth")
    checkpoint_output: str | None = Field(default=None)

    dataset: str | None = Field(default=None)
    datasets: list[str] | None = Field(default=None)
    dataset_split: str = Field(default="train")
    text_column: str = Field(default="text")
    dataset_format: str = Field(default="text")

    messages_column: str = Field(default="conversations")
    use_tokenizer_chat_template: bool = Field(default=False)
    use_unsloth_chat_template: bool = Field(default=True)
    chat_template_name: str = Field(default="chatml")
    chat_template_map_eos_token: bool = Field(default=False)
    strip_assistant_reasoning_blocks: bool = Field(default=True)
    role_key: str = Field(default="from")
    content_key: str = Field(default="value")
    user_roles: list[str] = Field(default=["human", "user"])
    assistant_roles: list[str] = Field(default=["gpt", "assistant", "model"])
    system_roles: list[str] = Field(default=["system"])

    load_in_4bit: bool = Field(default=False)
    bf16: bool = Field(default=True)
    fp16: bool = Field(default=False)
    max_seq_length: int = Field(default=4096)

    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=16)
    learning_rate: float = Field(default=2e-5)
    epochs: float = Field(default=1)
    max_steps: int = Field(default=-1)
    max_grad_norm: float = Field(default=1.0)
    warmup_steps: int = Field(default=5)
    warmup_ratio: float | None = Field(default=None)
    lr_scheduler_type: str = Field(default="constant")
    logging_steps: int = Field(default=1)
    save_steps: int = Field(default=50)

    lora_r: int = Field(default=16)
    lora_alpha: int = Field(default=16)
    lora_dropout: float = Field(default=0.0)
    bias: str = Field(default="none")
    use_gradient_checkpointing: bool | str = Field(default="unsloth")
    target_modules: list[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    merge_after: bool = Field(default=True)
    merged_save_method: str = Field(default="merged_16bit")
    packing: bool = Field(default=False)

    shuffle_dataset: bool = Field(default=True)
    seed: int = Field(default=3407)
    sample_size: int | None = Field(default=None)

    train_on_responses_only: bool = Field(default=False)
    instruction_part: str | None = Field(default=None)
    response_part: str | None = Field(default=None)
    strip_reasoning_tag_in_mask: bool = Field(default=True)

    wandb_enabled: bool = Field(default=False)
    wandb_project: str | None = Field(default=None)
    wandb_entity: str | None = Field(default=None)
    wandb_run_name: str | None = Field(default=None)
    wandb_tags: list[str] = Field(default_factory=list)


@dataclass
class UnslothStageResult:
    output_model: str
    checkpoint_dir: str
    latest_checkpoint: str | None
    completed: bool
    interrupted: bool


def _quantize_snapshot_to_4bit(model_name: str, output_model: str):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from .model import get_model_class

    def snapshot_is_reloadable(path: Path) -> bool:
        config_file = path / "config.json"
        if not config_file.exists():
            return False

        weight_files = list(path.glob("*.safetensors"))
        weight_files += list(path.glob("*.bin"))
        return len(weight_files) > 0

    output_path = Path(output_model)
    output_path.mkdir(parents=True, exist_ok=True)

    quant_model = None
    quant_tokenizer = None
    quant_errors = []

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    device_maps: list[str] = ["auto"]
    if not torch.cuda.is_available():
        device_maps = ["cpu"]
    else:
        device_maps.append("cpu")

    for index, device_map in enumerate(device_maps):
        try:
            quant_model = get_model_class(model_name).from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
            )

            try:
                quant_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    fix_mistral_regex=True,
                    trust_remote_code=True,
                )
            except TypeError:
                quant_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
            break
        except Exception as error:
            quant_errors.append((device_map, error))
            if index < len(device_maps) - 1:
                print(
                    f"[yellow]Quantization load with device_map='{device_map}' failed ({error}). Retrying...[/]"
                )

    if quant_model is None or quant_tokenizer is None:
        details = (
            "; ".join([f"{device_map}: {error}" for device_map, error in quant_errors])
            if quant_errors
            else "unknown error"
        )
        raise RuntimeError(
            "Failed to quantize merged model to 4-bit. "
            f"Details: {details}"
        )

    quant_model.save_pretrained(str(output_path))
    quant_tokenizer.save_pretrained(str(output_path))

    if not snapshot_is_reloadable(output_path):
        raise RuntimeError("4-bit export did not produce a reloadable snapshot")

    del quant_model, quant_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quantize_snapshot_to_4bit(model_name: str, output_model: str):
    _quantize_snapshot_to_4bit(model_name=model_name, output_model=output_model)


def _collect_lora_signal_stats(peft_model: Any) -> dict[str, float | int]:
    lora_a_tensors = 0
    lora_b_tensors = 0
    lora_b_nonzero_tensors = 0
    lora_b_abs_sum = 0.0
    lora_b_abs_max = 0.0

    for name, param in peft_model.named_parameters():
        if "lora_A" in name:
            lora_a_tensors += 1
        if "lora_B" not in name:
            continue

        lora_b_tensors += 1
        data = param.detach().float().cpu()
        abs_sum = float(data.abs().sum().item())
        abs_max = float(data.abs().max().item())
        lora_b_abs_sum += abs_sum
        lora_b_abs_max = max(lora_b_abs_max, abs_max)
        if abs_max > 0.0:
            lora_b_nonzero_tensors += 1

    return {
        "lora_a_tensors": lora_a_tensors,
        "lora_b_tensors": lora_b_tensors,
        "lora_b_nonzero_tensors": lora_b_nonzero_tensors,
        "lora_b_abs_sum": lora_b_abs_sum,
        "lora_b_abs_max": lora_b_abs_max,
    }


def _sample_lora_target_snapshots(
    peft_model: Any,
    max_params: int = 16,
) -> dict[str, torch.Tensor]:
    snapshots = {}

    for module_name, module in peft_model.named_modules():
        if len(snapshots) >= max_params:
            break
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if not hasattr(module, "base_layer") or not hasattr(module.base_layer, "weight"):
            continue

        weight = module.base_layer.weight
        if not isinstance(weight, torch.Tensor):
            continue
        if not torch.is_floating_point(weight):
            continue

        merged_weight_name = f"{module_name}.weight"
        snapshots[merged_weight_name] = weight.detach().float().cpu().clone()

    return snapshots


def _compare_parameter_deltas(
    merged_model: Any,
    snapshots: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if not snapshots:
        return {
            "compared_count": 0,
            "changed_count": 0,
            "max_mean_abs_delta": 0.0,
            "mean_abs_delta": 0.0,
            "samples": [],
        }

    merged_params = dict(merged_model.named_parameters())

    def resolve_param(name: str):
        after = merged_params.get(name)
        if after is not None:
            return after, name

        candidate_names = [name]
        for prefix in ["base_model.model.", "base_model."]:
            if name.startswith(prefix):
                candidate_names.append(name[len(prefix) :])

        for candidate in candidate_names:
            after = merged_params.get(candidate)
            if after is not None:
                return after, candidate

        # Last resort: unique suffix match for wrapped model naming differences.
        for candidate in candidate_names:
            suffix_matches = [
                merged_name
                for merged_name in merged_params
                if merged_name.endswith(candidate)
            ]
            if len(suffix_matches) == 1:
                matched_name = suffix_matches[0]
                return merged_params[matched_name], matched_name

        return None, None

    samples = []
    mean_deltas = []

    for name, before in snapshots.items():
        after, resolved_name = resolve_param(name)
        if after is None:
            continue

        after_cpu = after.detach().float().cpu()
        mean_abs_delta = float((after_cpu - before).abs().mean().item())
        label = resolved_name if resolved_name is not None else name
        samples.append((label, mean_abs_delta))
        mean_deltas.append(mean_abs_delta)

    compared_count = len(samples)
    changed_count = len([delta for _, delta in samples if delta > 1e-8])
    max_mean_abs_delta = max((delta for _, delta in samples), default=0.0)
    mean_abs_delta = float(sum(mean_deltas) / len(mean_deltas)) if mean_deltas else 0.0

    return {
        "compared_count": compared_count,
        "changed_count": changed_count,
        "max_mean_abs_delta": max_mean_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "samples": samples,
    }


def export_unsloth_checkpoint_snapshot(
    base_model: str,
    adapter_checkpoint: str,
    output_model: str,
    save_method: str,
    merge_base_model: str | None = None,
):
    base_model = str(base_model)
    adapter_checkpoint = str(adapter_checkpoint)
    if merge_base_model is not None:
        merge_base_model = str(merge_base_model)

    def snapshot_is_reloadable(path: Path) -> bool:
        config_file = path / "config.json"
        if not config_file.exists():
            return False

        weight_files = list(path.glob("*.safetensors"))
        weight_files += list(path.glob("*.bin"))
        return len(weight_files) > 0

    from peft import PeftModel
    from transformers import AutoTokenizer
    from transformers import PretrainedConfig
    from unsloth import FastLanguageModel
    from .model import get_model_class

    def load_export_tokenizer() -> Any:
        candidate_models: list[str] = []
        for candidate in [adapter_checkpoint, model_for_export, base_model]:
            if not candidate:
                continue
            if candidate in candidate_models:
                continue
            candidate_models.append(candidate)

        last_error = None
        for candidate in candidate_models:
            try:
                try:
                    return AutoTokenizer.from_pretrained(
                        candidate,
                        fix_mistral_regex=True,
                    )
                except TypeError:
                    return AutoTokenizer.from_pretrained(candidate)
            except Exception as error:
                last_error = error

        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not load tokenizer for checkpoint export")

    def is_prequantized_bnb4bit(model_name: str) -> bool:
        if not isinstance(model_name, str):
            model_name = str(model_name)

        try:
            config_dict, _ = PretrainedConfig.get_config_dict(model_name)
        except Exception:
            return False

        quant_config = config_dict.get("quantization_config")
        if not isinstance(quant_config, dict):
            return False
        return (
            quant_config.get("load_in_4bit") is True
            or quant_config.get("quant_method") == "bitsandbytes"
        )

    model_for_export = merge_base_model or base_model

    if save_method == "merged_16bit" and is_prequantized_bnb4bit(model_for_export):
        raise RuntimeError(
            "Cannot export merged 16-bit directly from a pre-quantized 4-bit base model. "
            "Provide a full-precision base model/path for 16-bit merge."
        )

    prequantized_export_base = is_prequantized_bnb4bit(model_for_export)

    load_in_4bit = save_method in ["merged_4bit", "forced_merged_4bit", "merged_4bit_forced"]
    if load_in_4bit and not prequantized_export_base:
        # If we're exporting 4-bit from a full-precision base, keep the model
        # in full precision and let save_pretrained_merged handle quantization.
        load_in_4bit = False

    output_path = Path(output_model)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_method == "merged_16bit":
        # For full-precision export, avoid Unsloth merge path and use a plain
        # Transformers + PEFT merge. This is more robust for adapter checkpoints.
        base = get_model_class(model_for_export).from_pretrained(
            model_for_export,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        peft_model = PeftModel.from_pretrained(
            base,
            adapter_checkpoint,
            is_trainable=False,
            device_map=None,
            low_cpu_mem_usage=False,
            autocast_adapter_dtype=False,
        )

        lora_signal = _collect_lora_signal_stats(peft_model)
        base_snapshots = _sample_lora_target_snapshots(peft_model, max_params=16)
        merged_model = peft_model.merge_and_unload()
        merge_deltas = _compare_parameter_deltas(merged_model, base_snapshots)

        tokenizer = load_export_tokenizer()

        merged_model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        verification = {
            "lora_signal": lora_signal,
            "merge_deltas": merge_deltas,
        }

        del merged_model, peft_model, base, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return verification

    model = None
    tokenizer = None

    load_errors = []
    load_attempts = ["cuda:0", "auto", "cpu"]
    for device_map in load_attempts:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_for_export,
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=load_in_4bit,
                device_map=device_map,
            )
            break
        except Exception as error:
            load_errors.append((device_map, error))

    if model is None or tokenizer is None:
        details = (
            "; ".join([f"{device_map}: {error}" for device_map, error in load_errors])
            if load_errors
            else "unknown error"
        )
        raise RuntimeError(
            "Failed to load base model for checkpoint export. "
            "This is usually a hardware/offload limitation for very large models. "
            "Try exporting on a machine with more VRAM/RAM, or keep using the adapter checkpoint directly. "
            f"Details: {details}"
        ) from (load_errors[-1][1] if load_errors else None)

    try:
        tokenizer = load_export_tokenizer()
    except Exception:
        # Keep tokenizer returned by FastLanguageModel as fallback.
        pass

    try:
        model = PeftModel.from_pretrained(
            model,
            adapter_checkpoint,
            is_trainable=False,
            device_map=None,
            low_cpu_mem_usage=False,
            autocast_adapter_dtype=False,
        )
    except TypeError:
        # Older PEFT versions may not support some kwargs.
        model = PeftModel.from_pretrained(
            model,
            adapter_checkpoint,
            is_trainable=False,
        )

    save_methods = [save_method]
    if save_method == "merged_4bit":
        if prequantized_export_base:
            save_methods = ["merged_4bit", "forced_merged_4bit", "merged_4bit_forced"]
        else:
            save_methods = ["forced_merged_4bit", "merged_4bit", "merged_4bit_forced"]

    last_error = None
    for method in save_methods:
        try:
            if hasattr(model, "save_pretrained_merged"):
                model.save_pretrained_merged(
                    str(output_path),
                    tokenizer,
                    save_method=method,
                )
            else:
                raise RuntimeError("save_pretrained_merged is not available on this adapter model")

            tokenizer.save_pretrained(str(output_path))
            if snapshot_is_reloadable(output_path):
                last_error = None
                break

            last_error = RuntimeError(f"Save method '{method}' did not produce a reloadable snapshot")
        except Exception as error:
            last_error = error

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if last_error is not None:
        raise RuntimeError(
            f"Failed to export checkpoint snapshot with methods {save_methods}."
        ) from last_error

    return None


def export_unsloth_checkpoint_snapshot_dual(
    base_model: str,
    adapter_checkpoint: str,
    output_model_16bit: str,
    output_model_4bit: str,
    merge_base_model: str | None = None,
):
    verification = export_unsloth_checkpoint_snapshot(
        base_model=base_model,
        adapter_checkpoint=adapter_checkpoint,
        output_model=output_model_16bit,
        save_method="merged_16bit",
        merge_base_model=merge_base_model,
    )

    # Quantize from the verified merged 16-bit snapshot to avoid
    # adapter re-attachment edge cases during combined export.
    _quantize_snapshot_to_4bit(
        model_name=output_model_16bit,
        output_model=output_model_4bit,
    )

    return verification


def run_unsloth_stage(
    settings: UnslothStageSettings,
    input_model: str,
    resume_from_checkpoint: str | None = None,
    export_snapshot: bool = True,
) -> UnslothStageResult:
    dataset_ids = []
    if settings.dataset is not None:
        dataset_ids.append(settings.dataset)
    if settings.datasets is not None:
        dataset_ids.extend(settings.datasets)

    if not dataset_ids:
        raise ValueError("Missing required field 'dataset' or 'datasets' in Unsloth config.")

    output_path = Path(settings.output_model)
    checkpoint_path = Path(
        settings.checkpoint_output or f"{settings.output_model}_checkpoints"
    )

    def find_latest_checkpoint(checkpoint_root: Path) -> str | None:
        checkpoints = []
        for entry in checkpoint_root.glob("checkpoint-*"):
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

    def export_model_snapshot():
        output_path.mkdir(parents=True, exist_ok=True)

        def snapshot_is_reloadable(path: Path) -> bool:
            config_file = path / "config.json"
            if not config_file.exists():
                return False

            weight_files = list(path.glob("*.safetensors"))
            weight_files += list(path.glob("*.bin"))
            return len(weight_files) > 0

        if settings.merge_after:
            if hasattr(model, "save_pretrained_merged"):
                save_methods = [settings.merged_save_method]
                if settings.merged_save_method == "merged_4bit":
                    # Some Unsloth versions expose one of these names.
                    save_methods = ["merged_4bit", "forced_merged_4bit", "merged_4bit_forced"]
                elif settings.merged_save_method == "merged_16bit":
                    # If 16-bit merge cannot produce a reloadable snapshot from a
                    # 4-bit base, try 4-bit forced merge variants as fallback.
                    save_methods = [
                        "merged_16bit",
                        "forced_merged_4bit",
                        "merged_4bit",
                        "merged_4bit_forced",
                    ]

                last_error = None
                for save_method in save_methods:
                    try:
                        model.save_pretrained_merged(
                            str(output_path),
                            tokenizer,
                            save_method=save_method,
                        )
                        tokenizer.save_pretrained(str(output_path))

                        if snapshot_is_reloadable(output_path):
                            last_error = None
                            break

                        last_error = RuntimeError(
                            f"Unsloth save_method '{save_method}' did not produce a reloadable snapshot."
                        )
                    except Exception as error:
                        last_error = error

                if last_error is not None:
                    raise RuntimeError(
                        f"Failed to save merged model using methods {save_methods}."
                    ) from last_error

            else:
                model.save_pretrained(str(output_path))
                tokenizer.save_pretrained(str(output_path))

            # Keep Qwen 3.5 linear-attention dtype settings consistent after merge.
            config_file = output_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as file:
                        config = json.load(file)

                    text_config = config.get("text_config")
                    if isinstance(text_config, dict):
                        model_dtype = str(config.get("torch_dtype", ""))
                        if (
                            model_dtype in {"bfloat16", "float16"}
                            and text_config.get("mamba_ssm_dtype") == "float32"
                        ):
                            text_config["mamba_ssm_dtype"] = model_dtype

                    # Training often stores use_cache=False for checkpointing,
                    # but inference speed is much better with KV cache enabled.
                    if config.get("use_cache") is False:
                        config["use_cache"] = True

                    with open(config_file, "w", encoding="utf-8") as file:
                        json.dump(config, file, indent=4)
                except Exception:
                    # Best effort only - do not fail the stage on config rewrite issues.
                    pass
        else:
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))

    report_to: str | list[str] = "none"
    if settings.wandb_enabled:
        try:
            import wandb  # ty:ignore[unresolved-import]

            del wandb
        except Exception as error:
            raise RuntimeError(
                "W&B logging enabled, but package is not installed. Install with `pip install wandb`."
            ) from error

        report_to = ["wandb"]
        if settings.wandb_project:
            os.environ["WANDB_PROJECT"] = settings.wandb_project
        if settings.wandb_entity:
            os.environ["WANDB_ENTITY"] = settings.wandb_entity
        if settings.wandb_run_name:
            os.environ["WANDB_NAME"] = settings.wandb_run_name
        if settings.wandb_tags:
            os.environ["WANDB_TAGS"] = ",".join(settings.wandb_tags)

    try:
        from datasets import load_dataset
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments

        try:
            from trl import SFTConfig
        except Exception:
            SFTConfig = None  # ty:ignore[invalid-assignment]
    except Exception as error:
        raise RuntimeError(
            "Missing Unsloth training dependencies. Install with e.g. `pip install unsloth trl`."
        ) from error

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=input_model,
        max_seq_length=settings.max_seq_length,
        dtype=None,
        load_in_4bit=settings.load_in_4bit,
    )

    peft_kwargs = {
        "r": settings.lora_r,
        "target_modules": settings.target_modules,
        "lora_alpha": settings.lora_alpha,
        "lora_dropout": settings.lora_dropout,
        "bias": settings.bias,
        "use_gradient_checkpointing": settings.use_gradient_checkpointing,
        "random_state": settings.seed,
    }

    try:
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    except TypeError:
        peft_kwargs.pop("random_state", None)
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    # Keep training memory footprint low regardless of backend defaults.
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    wants_checkpointing = settings.use_gradient_checkpointing not in [False, "false", "False"]
    if wants_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    loaded_datasets = []
    for index, dataset_id in enumerate(dataset_ids):
        dataset = load_dataset(dataset_id, split=settings.dataset_split)

        if settings.shuffle_dataset:
            dataset = dataset.shuffle(seed=settings.seed + index)

        if settings.sample_size is not None:
            sample_count = min(settings.sample_size, len(dataset))
            dataset = dataset.select(range(sample_count))

        loaded_datasets.append(dataset)

    if len(loaded_datasets) == 1:
        dataset = loaded_datasets[0]
    else:
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(loaded_datasets)

    if settings.dataset_format.lower() == "chatml" and settings.use_unsloth_chat_template:
        try:
            from unsloth.chat_templates import get_chat_template

            tokenizer = get_chat_template(
                tokenizer,
                chat_template=settings.chat_template_name,
                mapping={
                    "role": settings.role_key,
                    "content": settings.content_key,
                    "user": settings.user_roles[0] if settings.user_roles else "user",
                    "assistant": (
                        settings.assistant_roles[0]
                        if settings.assistant_roles
                        else "assistant"
                    ),
                },
                map_eos_token=settings.chat_template_map_eos_token,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to apply Unsloth chat template mapping. Check chat template and role/content mapping in stage config."
            ) from error

    user_roles = {role.lower() for role in settings.user_roles}
    assistant_roles = {role.lower() for role in settings.assistant_roles}
    system_roles = {role.lower() for role in settings.system_roles}

    def formatting_func(example: dict) -> list[str]:
        def strip_assistant_reasoning(text: str) -> str:
            if not settings.strip_assistant_reasoning_blocks:
                return text

            assistant_boundaries = [
                r"(<\|im_start\|>assistant\s*\n)",
                r"(<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\s*\n)",
            ]

            reasoning_wrappers = [
                (r"<think>", r"</think>"),
                (r"<thought>", r"</thought>"),
                (r"\[THINK\]", r"\[/THINK\]"),
            ]

            for boundary in assistant_boundaries:
                for start_tag, end_tag in reasoning_wrappers:
                    pattern = boundary + r"\s*" + start_tag + r".*?" + end_tag + r"\s*"
                    text = re.sub(
                        pattern,
                        r"\1",
                        text,
                        flags=re.IGNORECASE | re.DOTALL,
                    )

                # Also strip bare leading reasoning tags without a closing wrapper.
                for marker in [r"<think>", r"<thought>", r"\[THINK\]"]:
                    pattern = boundary + r"\s*" + marker + r"\s*"
                    text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

            return text

        if settings.dataset_format.lower() == "chatml":
            messages = example.get(settings.messages_column)
            if isinstance(messages, list):
                if hasattr(tokenizer, "apply_chat_template") and (
                    settings.use_unsloth_chat_template
                    or settings.use_tokenizer_chat_template
                ):
                    try:
                        formatted = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        if isinstance(formatted, str) and formatted.strip():
                            return [strip_assistant_reasoning(formatted)]
                    except Exception:
                        pass

                chat = []

                for message in messages:
                    if not isinstance(message, dict):
                        continue

                    raw_role = str(message.get(settings.role_key, "")).strip().lower()
                    if raw_role in user_roles:
                        role = "user"
                    elif raw_role in assistant_roles:
                        role = "assistant"
                    elif raw_role in system_roles:
                        role = "system"
                    else:
                        continue

                    content = str(message.get(settings.content_key, "")).strip()
                    if not content:
                        continue

                    chat.append({"role": role, "content": content})

                if chat:
                    if settings.use_tokenizer_chat_template and hasattr(
                        tokenizer,
                        "apply_chat_template",
                    ):
                        formatted = tokenizer.apply_chat_template(
                            chat,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        if isinstance(formatted, str):
                            return [formatted]

                    # Fallback if tokenizer has no chat template.
                    text = "".join(
                        [
                            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                            for message in chat
                        ]
                    )
                    return [strip_assistant_reasoning(text)]

        if settings.text_column in example:
            value = example[settings.text_column]
            if isinstance(value, list):
                return [str(v) for v in value]
            return [str(value)]

        if "text" in example:
            value = example["text"]
            if isinstance(value, list):
                return [str(v) for v in value]
            return [str(value)]

        if "prompt" in example and "completion" in example:
            return [f"{example['prompt']}\n{example['completion']}"]

        return [str(example)]

    formatted_text_column = "__heretic_text"

    def to_formatted_text(example: dict) -> dict[str, str]:
        formatted = formatting_func(example)
        text = formatted[0] if formatted else ""
        return {formatted_text_column: text}

    dataset = dataset.map(
        to_formatted_text,
        desc="Formatting training dataset",
    )
    dataset = dataset.filter(
        lambda row: isinstance(row[formatted_text_column], str)
        and len(row[formatted_text_column].strip()) > 0,
        desc="Filtering empty formatted samples",
    )

    if len(dataset) == 0:
        raise ValueError("No usable samples after formatting dataset.")

    sample_text = dataset[0][formatted_text_column]

    total_steps_for_warmup = settings.max_steps
    if total_steps_for_warmup is None or total_steps_for_warmup <= 0:
        total_steps_for_warmup = max(
            1,
            math.ceil(
                (len(dataset) * settings.epochs)
                / max(1, settings.batch_size * settings.gradient_accumulation_steps)
            ),
        )

    if settings.warmup_ratio is not None:
        effective_warmup_steps = max(1, int(total_steps_for_warmup * settings.warmup_ratio))
    else:
        effective_warmup_steps = settings.warmup_steps

    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        gradient_checkpointing=wants_checkpointing,
        warmup_steps=effective_warmup_steps,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.epochs,
        max_steps=settings.max_steps,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        optim="paged_adamw_8bit",
        max_grad_norm=settings.max_grad_norm,
        weight_decay=0.01,
        lr_scheduler_type=settings.lr_scheduler_type,
        seed=settings.seed,
        bf16=settings.bf16,
        fp16=settings.fp16,
        report_to=report_to,
        run_name=settings.wandb_run_name,
    )

    # TRL and Unsloth evolve quickly; build trainer kwargs dynamically to
    # support both old and new constructor signatures.
    signature = inspect.signature(SFTTrainer.__init__)
    param_names = set(signature.parameters.keys())

    args_for_trainer = training_args
    if SFTConfig is not None:
        sft_signature = inspect.signature(SFTConfig.__init__)
        sft_params = set(sft_signature.parameters.keys())
        if "dataset_text_field" in sft_params or "max_seq_length" in sft_params:
            sft_kwargs = {
                "output_dir": str(checkpoint_path),
                "per_device_train_batch_size": settings.batch_size,
                "gradient_accumulation_steps": settings.gradient_accumulation_steps,
                "gradient_checkpointing": wants_checkpointing,
                "warmup_steps": effective_warmup_steps,
                "learning_rate": settings.learning_rate,
                "num_train_epochs": settings.epochs,
                "max_steps": settings.max_steps,
                "logging_steps": settings.logging_steps,
                "save_steps": settings.save_steps,
                "optim": "paged_adamw_8bit",
                "max_grad_norm": settings.max_grad_norm,
                "weight_decay": 0.01,
                "lr_scheduler_type": settings.lr_scheduler_type,
                "seed": settings.seed,
                "bf16": settings.bf16,
                "fp16": settings.fp16,
            }
            if "report_to" in sft_params:
                sft_kwargs["report_to"] = report_to
            if "run_name" in sft_params and settings.wandb_run_name is not None:
                sft_kwargs["run_name"] = settings.wandb_run_name
            if "dataset_text_field" in sft_params:
                sft_kwargs["dataset_text_field"] = formatted_text_column
            if "max_seq_length" in sft_params:
                sft_kwargs["max_seq_length"] = settings.max_seq_length
            if "packing" in sft_params:
                sft_kwargs["packing"] = settings.packing
            args_for_trainer = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": args_for_trainer,
    }

    if "tokenizer" in param_names:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in param_names:
        trainer_kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in param_names:
        trainer_kwargs["dataset_text_field"] = formatted_text_column
    if "max_seq_length" in param_names:
        trainer_kwargs["max_seq_length"] = settings.max_seq_length
    if "packing" in param_names:
        trainer_kwargs["packing"] = settings.packing
    trainer = SFTTrainer(**trainer_kwargs)

    warmup_desc = (
        f"warmup_ratio={settings.warmup_ratio:.2%}"
        if settings.warmup_ratio is not None
        else f"warmup_steps={settings.warmup_steps}"
    )
    steps_desc = (
        f"max_steps={settings.max_steps}"
        if settings.max_steps and settings.max_steps > 0
        else f"epochs={settings.epochs}"
    )
    print(
        f"* Training config: learning_rate={settings.learning_rate:.2e}, scheduler={settings.lr_scheduler_type}, {steps_desc}, max_grad_norm={settings.max_grad_norm}, {warmup_desc}"
    )
    print(
        f"* Output paths: merged_or_adapter={output_path}, checkpoints={checkpoint_path}"
    )

    if settings.train_on_responses_only:
        try:
            from unsloth.chat_templates import train_on_responses_only
        except Exception as error:
            raise RuntimeError(
                "Response-only masking requested, but Unsloth chat template helpers are unavailable."
            ) from error

        def infer_mask_parts(sample_text: str) -> tuple[str | None, str | None]:
            candidates = [
                ("<|im_start|>user\n", "<|im_start|>assistant\n"),
                ("<|im_start|>user", "<|im_start|>assistant"),
                (
                    "<|start_header_id|>user<|end_header_id|>\n\n",
                    "<|start_header_id|>assistant<|end_header_id|>\n\n",
                ),
                (
                    "<|start_header_id|>user<|end_header_id|>",
                    "<|start_header_id|>assistant<|end_header_id|>",
                ),
                ("[INST]", "[/INST]"),
            ]

            for instruction, response in candidates:
                if instruction in sample_text and response in sample_text:
                    return instruction, response

            # Generic fallback for ChatML-like role tags.
            user_match = re.search(r"<\|im_start\|>user\n?", sample_text)
            assistant_match = re.search(
                r"<\|im_start\|>assistant\n?",
                sample_text,
            )
            if user_match and assistant_match:
                return user_match.group(0), assistant_match.group(0)

            return None, None

        instruction_part = settings.instruction_part
        response_part = settings.response_part

        def normalize_response_part(value: str | None) -> str | None:
            if value is None:
                return None

            if not settings.strip_reasoning_tag_in_mask:
                return value

            for marker in ["<think>", "<thought>", "[THINK]"]:
                index = value.find(marker)
                if index >= 0:
                    return value[:index]

            return value

        if instruction_part is None or response_part is None:
            inferred_instruction, inferred_response = infer_mask_parts(sample_text)

            if instruction_part is None:
                instruction_part = inferred_instruction
            if response_part is None:
                response_part = inferred_response

            # Prefer stable role-boundary markers over reasoning-tag markers.
            if instruction_part is None and settings.dataset_format.lower() == "chatml":
                instruction_part = "<|im_start|>user\n"
            if response_part is None and settings.dataset_format.lower() == "chatml":
                response_part = "<|im_start|>assistant\n"

            response_part = normalize_response_part(response_part)

            if instruction_part is None or response_part is None:
                raise ValueError(
                    "Could not infer response-only masking delimiters. Set instruction_part and response_part explicitly in the stage config."
                )

        response_part = normalize_response_part(response_part)

        # Show resolved masking markers to make debugging easier.
        print(
            f"* Response-only masking: instruction_part={instruction_part!r}, response_part={response_part!r}"
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    interrupted = False
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[yellow]Training interrupted by user.[/]")
        trainer.save_state()

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = find_latest_checkpoint(checkpoint_path)

    resolved_output_model = str(output_path)

    # Prefer checkpoint output for interactive stage handling.
    if latest_checkpoint is not None:
        resolved_output_model = latest_checkpoint
        if interrupted:
            print(
                f"* Using latest adapter checkpoint [bold]{latest_checkpoint}[/] as interrupted snapshot"
            )

    if export_snapshot:
        if interrupted:
            # Do not force a merge/export when the user interrupts training.
            # Keep working from checkpoint artifacts.
            pass
        else:
            try:
                export_model_snapshot()
                resolved_output_model = str(output_path)
            except Exception as error:
                if latest_checkpoint is not None:
                    print(
                        f"[yellow]* Snapshot export failed ({error}). Falling back to latest adapter checkpoint [bold]{latest_checkpoint}[/].[/]"
                    )
                    resolved_output_model = latest_checkpoint
                else:
                    raise

    # Ensure references are released before downstream model reload.
    del trainer, model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return UnslothStageResult(
        output_model=resolved_output_model,
        checkpoint_dir=str(checkpoint_path),
        latest_checkpoint=latest_checkpoint,
        completed=not interrupted,
        interrupted=interrupted,
    )
