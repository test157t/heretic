# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import inspect
import json
import math
import os
from pathlib import Path
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from . import _warnings as _heretic_warnings  # noqa: F401

import torch
from pydantic import BaseModel, Field

# NVFP4 (E2M1) representable positive values.
# Format: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit.
_NVFP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_NVFP4_MAX = 6.0


def _quantize_nvfp4(tensor: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Per-group NVFP4 (E2M1) fake quantization — quantize and dequantize."""
    orig_shape = tensor.shape
    flat = tensor.flatten()
    pad = (group_size - flat.numel() % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    groups = flat.view(-1, group_size)
    max_abs = groups.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    scale = max_abs / _NVFP4_MAX
    scaled = groups / scale
    abs_scaled = scaled.abs()
    sign = scaled.sign()
    values = _NVFP4_VALUES.to(tensor.device)
    indices = torch.bucketize(abs_scaled, (values[1:] + values[:-1]) / 2)
    indices = indices.clamp(0, len(values) - 1)
    quantized = values[indices] * sign * scale
    if pad:
        quantized = quantized.flatten()[:-pad]
    return quantized.reshape(orig_shape)


def _constrain_weights_to_nvfp4(model: torch.nn.Module, group_size: int = 128) -> None:
    """Project all nn.Linear weights onto the NVFP4 manifold in-place."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight"):
            orig_data = module.weight.data
            module.weight.data = _quantize_nvfp4(orig_data, group_size)


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
    role_key: str = Field(default="from")
    content_key: str = Field(default="value")
    user_roles: list[str] = Field(default=["human", "user"])
    assistant_roles: list[str] = Field(default=["gpt", "assistant", "model"])
    system_roles: list[str] = Field(default=["system"])

    load_in_4bit: bool = Field(default=False)
    device_map: str | dict[str, int | str] = Field(default="cuda:0")
    max_memory: dict[str, str] | None = Field(default=None)
    bf16: bool = Field(default=True)
    fp16: bool = Field(default=False)
    auto_mixed_precision: bool = Field(default=True)
    allow_tf32: bool = Field(default=True)
    max_seq_length: int = Field(default=4096)
    chat_max_seq_length: int | None = Field(default=2048)
    chat_max_response_length: int = Field(default=1024)
    attention_backend: str = Field(default="auto")

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
    modules_to_save: list[str] | None = Field(default=None)

    merge_after: bool = Field(default=True)
    merged_save_method: str = Field(default="merged_16bit")
    packing: bool = Field(default=False)
    padding_free: bool | None = Field(default=None)

    shuffle_dataset: bool = Field(default=True)
    seed: int = Field(default=3407)
    sample_size: int | None = Field(default=None)
    eval_split_percent: float = Field(default=0.0)
    eval_steps: int | None = Field(default=None)
    eval_batch_size: int = Field(default=1)
    eval_accumulation_steps: int | None = Field(default=None)
    chunk_eval_across_train: bool = Field(default=False)
    eval_sample_strategy: str = Field(default="fixed")
    eval_subset_percent: float | None = Field(default=None)
    eval_samples_per_run: int | None = Field(default=None)

    train_on_responses_only: bool = Field(default=False)
    instruction_part: str | None = Field(default=None)
    response_part: str | None = Field(default=None)
    group_by_length: bool = Field(default=False)
    length_column_name: str = Field(default="__heretic_length")
    preprocessing_batch_size: int = Field(default=1000)
    dataset_num_proc: int | None = Field(default=None)
    dataloader_num_workers: int = Field(default=0)
    dataloader_pin_memory: bool = Field(default=True)

    memory_management_enabled: bool = Field(default=False)
    empty_cache_before_save_eval: bool = Field(default=True)
    memory_defrag_every_n_steps: int = Field(default=10)
    memory_defrag_reserved_minus_alloc_gb: float = Field(default=4.0)
    memory_defrag_target_fraction: float = Field(default=0.90)
    memory_debug_prints: bool = Field(default=False)
    per_process_vram_fraction_cap: float | None = Field(default=None)

    qat_enabled: bool = Field(default=False)
    qat_scheme: str = Field(default="int4")

    nvfp4_emulation: bool = Field(default=False)
    nvfp4_group_size: int = Field(default=128)

    wandb_enabled: bool = Field(default=False)
    wandb_project: str | None = Field(default=None)
    wandb_entity: str | None = Field(default=None)
    wandb_run_name: str | None = Field(default=None)
    wandb_tags: list[str] = Field(default_factory=list)


@contextmanager
def _patch_peft_for_gemma4_clippable_linear():
    try:
        from peft.tuners.lora.model import LoraModel
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        yield
        return

    original_create_and_replace = LoraModel._create_and_replace
    if getattr(original_create_and_replace, "_heretic_gemma4_clippable_patch", False):
        yield
        return

    def patched_create_and_replace(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key=None,
        **kwargs,
    ):
        if isinstance(target, Gemma4ClippableLinear):
            return original_create_and_replace(
                self,
                peft_config,
                adapter_name,
                target.linear,
                "linear",
                target,
                current_key=current_key,
                **kwargs,
            )
        return original_create_and_replace(
            self,
            peft_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key=current_key,
            **kwargs,
        )

    patched_create_and_replace._heretic_gemma4_clippable_patch = True
    LoraModel._create_and_replace = patched_create_and_replace
    try:
        yield
    finally:
        if LoraModel._create_and_replace is patched_create_and_replace:
            LoraModel._create_and_replace = original_create_and_replace


@dataclass
class UnslothStageResult:
    output_model: str
    checkpoint_dir: str
    latest_checkpoint: str | None
    completed: bool
    interrupted: bool


def _prepare_unsloth_import() -> None:
    # On Windows, recent Unsloth can crash if its dataset patch imports
    # pyarrow after Unsloth has already started patching. Importing datasets
    # first matches the training path and avoids that native crash.
    import datasets  # noqa: F401


class UnslothCheckpointChatModel:
    def __init__(
        self,
        settings: UnslothStageSettings,
        base_model: str,
        adapter_checkpoint: str | None = None,
    ):
        _prepare_unsloth_import()
        from unsloth import FastLanguageModel
        from peft import PeftModel
        from transformers import TextStreamer

        self.settings = settings
        self.TextStreamer = TextStreamer

        if settings.per_process_vram_fraction_cap is not None and torch.cuda.is_available():
            if not 0 < settings.per_process_vram_fraction_cap <= 1:
                raise ValueError("per_process_vram_fraction_cap must be > 0 and <= 1.")
            torch.cuda.set_per_process_memory_fraction(
                settings.per_process_vram_fraction_cap,
            )
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(
                f"* CUDA allocator cap: [bold]{settings.per_process_vram_fraction_cap * total_gb:.2f} GB[/] "
                f"({settings.per_process_vram_fraction_cap:.0%} of GPU VRAM)"
            )

        def normalize_device_map(device_map: str | dict[str, int | str]):
            if device_map in {"cuda", "cuda:0"}:
                return {"": 0}
            return device_map

        def print_cuda_memory(label: str):
            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(
                f"* CUDA memory {label}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
            )

        model_kwargs = {
            "model_name": base_model,
            "max_seq_length": settings.chat_max_seq_length or settings.max_seq_length,
            "dtype": None,
            "load_in_4bit": settings.load_in_4bit,
            "device_map": normalize_device_map(settings.device_map),
        }
        if settings.max_memory is not None:
            model_kwargs["max_memory"] = {
                int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()
            }
        print(f"* Device map: [bold]{model_kwargs['device_map']}[/]")
        attention_backend = settings.attention_backend.lower()
        if attention_backend == "fa2":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif attention_backend == "xformers":
            print(
                "* Attention backend: xformers requested; using Unsloth auto-selection"
            )

        print_cuda_memory("before base load")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                **model_kwargs
            )
        except (TypeError, ValueError) as error:
            if (
                "attn_implementation" not in model_kwargs
                or "attn_implementation" not in str(error)
            ):
                raise
            print(
                f"[yellow]* Attention backend override rejected ({error}). Falling back to Unsloth auto-selection.[/]"
            )
            model_kwargs.pop("attn_implementation")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                **model_kwargs
            )
        print_cuda_memory("after base load")

        if adapter_checkpoint is not None:
            print(
                f"* Loading adapter checkpoint [bold]{adapter_checkpoint}[/] with Unsloth... ",
                end="",
            )
            try:
                with _patch_peft_for_gemma4_clippable_linear():
                    try:
                        self.model = PeftModel.from_pretrained(
                            self.model,
                            adapter_checkpoint,
                            is_trainable=False,
                            device_map=None,
                            low_cpu_mem_usage=False,
                            autocast_adapter_dtype=False,
                        )
                    except TypeError:
                        self.model = PeftModel.from_pretrained(
                            self.model,
                            adapter_checkpoint,
                            is_trainable=False,
                        )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    "CUDA out of memory while loading the LoRA adapter for checkpoint chat. "
                    "The base model already loaded successfully, so reduce chat_max_seq_length, "
                    "use load_in_4bit, or save/export the checkpoint instead of chatting with it in-process."
                ) from error
            print("[green]Ok[/]")
            self._print_loaded_adapter_signal()
            print_cuda_memory("after adapter load")

        if hasattr(FastLanguageModel, "for_inference"):
            try:
                FastLanguageModel.for_inference(self.model)
                print_cuda_memory("after inference prep")
            except Exception as error:
                print(
                    f"[yellow]* Unsloth inference optimization failed ({error}). Falling back to standard eval mode.[/]"
                )
                self.model.eval()

    def _print_loaded_adapter_signal(self):
        lora_tensors = 0
        nonzero_tensors = 0
        abs_sum = 0.0
        abs_max = 0.0
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                continue

            lora_tensors += 1
            data = param.detach().float()
            tensor_abs_sum = float(data.abs().sum().item())
            tensor_abs_max = float(data.abs().max().item()) if data.numel() > 0 else 0.0
            abs_sum += tensor_abs_sum
            abs_max = max(abs_max, tensor_abs_max)
            if tensor_abs_sum > 0.0:
                nonzero_tensors += 1

        print(
            f"* Loaded adapter signal: [bold]{nonzero_tensors}[/]/[bold]{lora_tensors}[/] LoRA tensors nonzero (abs_sum={abs_sum:.4e}, max={abs_max:.4e})"
        )
        if lora_tensors == 0 or abs_sum <= 0.0:
            print(
                "[yellow]* Warning: loaded adapter appears to have no LoRA signal.[/]"
            )

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat_prompt = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)
        streamer = self.TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.settings.chat_max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )


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
            f"Failed to quantize merged model to 4-bit. Details: {details}"
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
        if not hasattr(module, "base_layer") or not hasattr(
            module.base_layer, "weight"
        ):
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
    qat_scheme: str | None = None,
):
    _prepare_unsloth_import()
    from unsloth import FastLanguageModel
    from pathlib import Path
    from peft import PeftModel
    from transformers import AutoTokenizer
    from transformers import PretrainedConfig
    from .model import get_model_class

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

    def load_export_tokenizer() -> Any:
        # Try to load tokenizer from checkpoint directory first
        # The checkpoint directory should contain the tokenizer with the chat template
        checkpoint_dir = Path(adapter_checkpoint)
        checkpoint_tokenizer = None
        if checkpoint_dir.exists():
            try:
                try:
                    checkpoint_tokenizer = AutoTokenizer.from_pretrained(
                        str(checkpoint_dir),
                        fix_mistral_regex=True,
                    )
                except TypeError:
                    checkpoint_tokenizer = AutoTokenizer.from_pretrained(
                        str(checkpoint_dir),
                    )
                print(f"* Loaded tokenizer from checkpoint directory: {checkpoint_dir}")
                return checkpoint_tokenizer
            except Exception as e:
                print(
                    f"[yellow]* Could not load tokenizer from checkpoint directory: {e}[/]"
                )

        # Try parent directory as fallback
        parent_dir = Path(adapter_checkpoint).parent
        if parent_dir.exists():
            try:
                try:
                    checkpoint_tokenizer = AutoTokenizer.from_pretrained(
                        str(parent_dir),
                        fix_mistral_regex=True,
                    )
                except TypeError:
                    checkpoint_tokenizer = AutoTokenizer.from_pretrained(
                        str(parent_dir),
                    )
                print(f"* Loaded tokenizer from parent directory: {parent_dir}")
                return checkpoint_tokenizer
            except Exception as e:
                print(
                    f"[yellow]* Could not load tokenizer from parent directory: {e}[/]"
                )

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
                    tokenizer = AutoTokenizer.from_pretrained(
                        candidate,
                        fix_mistral_regex=True,
                    )
                except TypeError:
                    tokenizer = AutoTokenizer.from_pretrained(candidate)

                # Ensure chat template is loaded from the tokenizer's configuration
                # The chat_template attribute should be set from the tokenizer_config.json
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    tokenizer_config_path = candidate_path / "tokenizer_config.json"
                    if tokenizer_config_path.exists():
                        try:
                            with open(
                                tokenizer_config_path, "r", encoding="utf-8"
                            ) as f:
                                config = json.load(f)
                                if "chat_template" in config:
                                    tokenizer.chat_template = config["chat_template"]
                                    # Debug: print chat template status
                                    template_preview = (
                                        str(config["chat_template"])[:100] + "..."
                                        if len(str(config["chat_template"])) > 100
                                        else str(config["chat_template"])
                                    )
                                    print(
                                        f"* Loaded chat template from {candidate}: {template_preview}"
                                    )
                        except Exception as e:
                            print(
                                f"[yellow]* Warning: could not read tokenizer_config.json from {candidate}: {e}[/]"
                            )

                return tokenizer
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

    load_in_4bit = save_method in [
        "merged_4bit",
        "forced_merged_4bit",
        "merged_4bit_forced",
    ]
    if load_in_4bit and not prequantized_export_base:
        # If we're exporting 4-bit from a full-precision base, keep the model
        # in full precision and let save_pretrained_merged handle quantization.
        load_in_4bit = False

    output_path = Path(output_model)
    output_path.mkdir(parents=True, exist_ok=True)

    is_qat_export = save_method.startswith("qat_")
    if is_qat_export:
        if qat_scheme is None:
            qat_scheme = save_method.replace("qat_", "")
            if qat_scheme not in {"int4", "int8", "fp8"}:
                qat_scheme = "int4"

        qat_scheme_full = qat_scheme
        if qat_scheme == "int4":
            qat_scheme_full = "int4"
        elif qat_scheme == "int8":
            qat_scheme_full = "int8-int4"
        elif qat_scheme == "fp8":
            qat_scheme_full = "fp8-int4"

        print(f"* Loading base model with Unsloth for QAT ({qat_scheme_full}) export...")
        qat_model = None
        qat_tokenizer = None
        load_errors = []
        load_attempts = ["cuda:0", "auto", "cpu"]
        for device_map in load_attempts:
            try:
                qat_model, qat_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_for_export,
                    max_seq_length=4096,
                    dtype=None,
                    load_in_4bit=False,
                    device_map=device_map,
                )
                break
            except Exception as error:
                load_errors.append((device_map, error))

        if qat_model is None or qat_tokenizer is None:
            details = "; ".join(
                f"{device_map}: {error}" for device_map, error in load_errors
            )
            raise RuntimeError(
                f"Failed to load base model for QAT export. Details: {details}"
            )

        qat_model = FastLanguageModel.get_peft_model(
            qat_model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,
            qat_scheme=qat_scheme_full,
        )

        with _patch_peft_for_gemma4_clippable_linear():
            try:
                qat_model = PeftModel.from_pretrained(
                    qat_model,
                    adapter_checkpoint,
                    is_trainable=False,
                    device_map=None,
                    low_cpu_mem_usage=False,
                    autocast_adapter_dtype=False,
                )
            except TypeError:
                qat_model = PeftModel.from_pretrained(
                    qat_model,
                    adapter_checkpoint,
                    is_trainable=False,
                )

        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig

        quantize_(qat_model, QATConfig(step="convert"))

        try:
            qat_tokenizer = load_export_tokenizer()
        except Exception:
            pass

        if hasattr(qat_model, "save_pretrained_torchao"):
            qat_model.save_pretrained_torchao(
                str(output_path),
                qat_tokenizer,
                torchao_config=qat_model._torchao_config.base_config,
            )
        else:
            raise RuntimeError(
                "save_pretrained_torchao is not available. Install torchao."
            )

        qat_tokenizer.save_pretrained(str(output_path))
        del qat_model, qat_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    if save_method == "merged_16bit":
        # For full-precision export, avoid Unsloth merge path and use a plain
        # Transformers + PEFT merge. This is more robust for adapter checkpoints.
        base = get_model_class(model_for_export).from_pretrained(
            model_for_export,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        with _patch_peft_for_gemma4_clippable_linear():
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

    with _patch_peft_for_gemma4_clippable_linear():
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
                raise RuntimeError(
                    "save_pretrained_merged is not available on this adapter model"
                )

            tokenizer.save_pretrained(str(output_path))
            if snapshot_is_reloadable(output_path):
                last_error = None
                break

            last_error = RuntimeError(
                f"Save method '{method}' did not produce a reloadable snapshot"
            )
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
        raise ValueError(
            "Missing required field 'dataset' or 'datasets' in Unsloth config."
        )

    def normalize_device_map(device_map: str | dict[str, int | str]):
        if device_map in {"cuda", "cuda:0"}:
            return {"": 0}
        return device_map

    output_path = Path(settings.output_model)
    checkpoint_path = Path(
        settings.checkpoint_output or f"{settings.output_model}_checkpoints"
    )
    model = None
    tokenizer = None

    if settings.per_process_vram_fraction_cap is not None and torch.cuda.is_available():
        if not 0 < settings.per_process_vram_fraction_cap <= 1:
            raise ValueError("per_process_vram_fraction_cap must be > 0 and <= 1.")
        torch.cuda.set_per_process_memory_fraction(
            settings.per_process_vram_fraction_cap,
        )
        print(
            f"* CUDA per-process memory cap: {settings.per_process_vram_fraction_cap:.0%}"
        )

    if settings.preprocessing_batch_size <= 0:
        raise ValueError("preprocessing_batch_size must be greater than 0.")
    if settings.dataset_num_proc is not None and settings.dataset_num_proc <= 0:
        raise ValueError("dataset_num_proc must be greater than 0 when set.")
    if settings.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be greater than or equal to 0.")
    if settings.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be greater than 0.")
    if settings.eval_accumulation_steps is not None and settings.eval_accumulation_steps <= 0:
        raise ValueError("eval_accumulation_steps must be greater than 0 when set.")
    if settings.eval_samples_per_run is not None and settings.eval_samples_per_run <= 0:
        raise ValueError("eval_samples_per_run must be greater than 0 when set.")
    if settings.eval_subset_percent is not None and not 0 < settings.eval_subset_percent <= 100:
        raise ValueError("eval_subset_percent must be > 0 and <= 100 when set.")
    eval_sample_strategy = settings.eval_sample_strategy.lower()
    if eval_sample_strategy not in {"full", "fixed", "chunked"}:
        raise ValueError('eval_sample_strategy must be "full", "fixed", or "chunked".')
    if settings.chunk_eval_across_train:
        eval_sample_strategy = "chunked"

    effective_bf16 = settings.bf16
    effective_fp16 = settings.fp16
    if settings.auto_mixed_precision and not effective_bf16 and not effective_fp16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            effective_bf16 = True
        elif torch.cuda.is_available():
            effective_fp16 = True

    if effective_bf16 and effective_fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled.")

    if settings.qat_enabled:
        valid_qat_schemes = {"fp8-int4", "fp8-fp8", "int8-int4", "int4"}
        if settings.qat_scheme not in valid_qat_schemes:
            raise ValueError(
                f"qat_scheme must be one of {valid_qat_schemes}, got '{settings.qat_scheme}'."
            )
        if settings.load_in_4bit:
            print(
                "[yellow]* QAT requires 16-bit base model; overriding load_in_4bit to False.[/]"
            )

    if settings.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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

    def export_model_snapshot(model: Any, tokenizer: Any):
        output_path.mkdir(parents=True, exist_ok=True)

        def snapshot_is_reloadable(path: Path) -> bool:
            config_file = path / "config.json"
            if not config_file.exists():
                return False

            weight_files = list(path.glob("*.safetensors"))
            weight_files += list(path.glob("*.bin"))
            return len(weight_files) > 0

        def finalize_config(config_dir: Path):
            config_file = config_dir / "config.json"
            if not config_file.exists():
                return

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

                if config.get("use_cache") is False:
                    config["use_cache"] = True

                with open(config_file, "w", encoding="utf-8") as file:
                    json.dump(config, file, indent=4)
            except Exception:
                pass

        is_qat = getattr(model, "_torchao_config", None) is not None

        if is_qat:
            print("* Exporting QAT model via torchao...")
            try:
                from torchao.quantization import quantize_
                from torchao.quantization.qat import QATConfig
            except ImportError:
                raise RuntimeError(
                    "torchao is required for QAT export. Install with `pip install torchao`."
                )

            quantize_(model, QATConfig(step="convert"))

            if hasattr(model, "save_pretrained_torchao"):
                model.save_pretrained_torchao(
                    str(output_path),
                    tokenizer,
                    torchao_config=model._torchao_config.base_config,
                )
            else:
                raise RuntimeError(
                    "save_pretrained_torchao is not available on this Unsloth model."
                )

            tokenizer.save_pretrained(str(output_path))
            finalize_config(output_path)
            return

        if settings.merge_after:
            if hasattr(model, "save_pretrained_merged"):
                save_methods = [settings.merged_save_method]
                if settings.merged_save_method == "merged_4bit":
                    save_methods = [
                        "merged_4bit",
                        "forced_merged_4bit",
                        "merged_4bit_forced",
                    ]
                elif settings.merged_save_method == "merged_16bit":
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

                        if (
                            hasattr(tokenizer, "chat_template")
                            and tokenizer.chat_template
                        ):
                            pass
                        else:
                            try:
                                candidate_path = Path(input_model)
                                if candidate_path.exists():
                                    tokenizer_config_path = (
                                        candidate_path / "tokenizer_config.json"
                                    )
                                    if tokenizer_config_path.exists():
                                        with open(
                                            tokenizer_config_path, "r", encoding="utf-8"
                                        ) as f:
                                            tconfig = json.load(f)
                                            if "chat_template" in tconfig:
                                                tokenizer.chat_template = tconfig[
                                                    "chat_template"
                                                ]
                            except Exception:
                                pass

                        output_path_obj = Path(output_path)
                        tokenizer_config_path = (
                            output_path_obj / "tokenizer_config.json"
                        )
                        if tokenizer_config_path.exists():
                            try:
                                with open(
                                    tokenizer_config_path, "r", encoding="utf-8"
                                ) as f:
                                    tconfig = json.load(f)
                                    if "chat_template" in tconfig:
                                        template_preview = (
                                            str(tconfig["chat_template"])[:100] + "..."
                                            if len(str(tconfig["chat_template"])) > 100
                                            else str(tconfig["chat_template"])
                                        )
                                        print(
                                            f"* Chat template saved: {template_preview}"
                                        )
                                    else:
                                        print(
                                            "[yellow]* Warning: chat_template not found in tokenizer_config.json[/]"
                                        )
                            except Exception as e:
                                print(
                                    f"[yellow]* Warning: could not read tokenizer_config.json: {e}[/]"
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

            finalize_config(output_path)
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
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

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

    effective_load_in_4bit = settings.load_in_4bit
    if settings.qat_enabled:
        effective_load_in_4bit = False

    model_kwargs = {
        "model_name": input_model,
        "max_seq_length": settings.max_seq_length,
        "dtype": None,
        "device_map": normalize_device_map(settings.device_map),
    }
    if effective_load_in_4bit:
        _quantize_lm_head = "lm_head" in settings.target_modules
        if _quantize_lm_head:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_skip_modules=["embed_tokens"],
            )
        else:
            model_kwargs["load_in_4bit"] = True
    if settings.max_memory is not None:
        model_kwargs["max_memory"] = {
            int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()
        }
    print(f"* Device map: [bold]{model_kwargs['device_map']}[/]")
    attention_backend = settings.attention_backend.lower()
    if attention_backend not in {"auto", "xformers", "fa2"}:
        raise ValueError('attention_backend must be "auto", "xformers", or "fa2".')
    if attention_backend == "fa2":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif attention_backend == "xformers":
        # Transformers does not expose xformers as an attn_implementation value.
        # Unsloth detects and applies its xformers path internally when available.
        print("* Attention backend: xformers requested; using Unsloth auto-selection")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    except (TypeError, ValueError) as error:
        if (
            "attn_implementation" not in model_kwargs
            or "attn_implementation" not in str(error)
        ):
            raise
        print(
            f"[yellow]* Attention backend override rejected ({error}). Falling back to Unsloth auto-selection.[/]"
        )
        model_kwargs.pop("attn_implementation")
        model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
        jinja_path = Path(input_model) / "chat_template.jinja"
        if jinja_path.exists():
            try:
                template = jinja_path.read_text(encoding="utf-8")
                tokenizer.chat_template = template
                print(f"* Loaded chat template from {jinja_path}")
            except Exception as error:
                print(
                    f"[yellow]* Warning: could not read chat_template.jinja from {jinja_path}: {error}[/]"
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
    if settings.modules_to_save:
        peft_kwargs["modules_to_save"] = settings.modules_to_save
    if settings.qat_enabled:
        peft_kwargs["qat_scheme"] = settings.qat_scheme

    try:
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    except TypeError:
        peft_kwargs.pop("random_state", None)
        if settings.qat_enabled:
            peft_kwargs.pop("qat_scheme", None)
            print(
                "[yellow]* Unsloth version does not support qat_scheme; disabling QAT.[/]"
            )
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    if settings.nvfp4_emulation:
        _constrain_weights_to_nvfp4(model, settings.nvfp4_group_size)
        print(
            f"* NVFP4 emulation enabled: weights constrained to E2M1 format "
            f"(group_size={settings.nvfp4_group_size})"
        )

    # Keep training memory footprint low regardless of backend defaults.
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    wants_checkpointing = settings.use_gradient_checkpointing not in [
        False,
        "false",
        "False",
    ]
    if wants_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
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

    if not 0 <= settings.eval_split_percent < 100:
        raise ValueError("eval_split_percent must be >= 0 and < 100.")

    eval_dataset = None
    if settings.eval_split_percent > 0:
        split_dataset = dataset.train_test_split(
            test_size=settings.eval_split_percent / 100,
            seed=settings.seed,
            shuffle=True,
        )
        dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    if (
        settings.dataset_format.lower() == "chatml"
        and settings.use_unsloth_chat_template
    ):
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

            # Ensure the chat template is properly saved with the tokenizer
            # The get_chat_template function modifies the tokenizer's chat_template attribute
            # but we need to make sure it's persisted
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                # Chat template is set, this should be saved with the tokenizer
                pass
            else:
                # Fallback: try to set the chat template manually
                try:
                    from unsloth.chat_templates import get_template

                    template = get_template(settings.chat_template_name)
                    if template:
                        tokenizer.chat_template = template
                except Exception:
                    pass

            # Debug: print chat template status
            if hasattr(tokenizer, "chat_template"):
                template_preview = (
                    str(tokenizer.chat_template)[:100] + "..."
                    if len(str(tokenizer.chat_template)) > 100
                    else str(tokenizer.chat_template)
                )
                print(f"* Chat template set: {template_preview}")
        except Exception as error:
            raise RuntimeError(
                "Failed to apply Unsloth chat template mapping. Check chat template and role/content mapping in stage config."
            ) from error

    user_roles = {role.lower() for role in settings.user_roles}
    assistant_roles = {role.lower() for role in settings.assistant_roles}
    system_roles = {role.lower() for role in settings.system_roles}

    def formatting_func(example: dict, tokenizer: Any) -> list[str]:
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
                            return [formatted]
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
                    return [text]

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

    def to_formatted_text(
        example: dict, tokenizer: Any = tokenizer
    ) -> dict[str, str]:
        formatted = formatting_func(example, tokenizer)
        text = formatted[0] if formatted else ""
        return {formatted_text_column: text}

    def to_formatted_text_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        sample_count = len(next(iter(batch.values()), []))
        texts = []
        for index in range(sample_count):
            example = {key: values[index] for key, values in batch.items()}
            texts.append(to_formatted_text(example)[formatted_text_column])
        return {formatted_text_column: texts}

    def format_and_filter_dataset(dataset_to_prepare: Any, description: str) -> Any:
        prepared_dataset = dataset_to_prepare.map(
            to_formatted_text_batch,
            batched=True,
            batch_size=settings.preprocessing_batch_size,
            num_proc=settings.dataset_num_proc,
            desc=f"Formatting {description} dataset",
        )
        return prepared_dataset.filter(
            lambda row: (
                isinstance(row[formatted_text_column], str)
                and len(row[formatted_text_column].strip()) > 0
            ),
            num_proc=settings.dataset_num_proc,
            desc=f"Filtering empty {description} samples",
        )

    dataset = format_and_filter_dataset(dataset, "training")
    if eval_dataset is not None:
        eval_dataset = format_and_filter_dataset(eval_dataset, "evaluation")

    if len(dataset) == 0:
        raise ValueError("No usable samples after formatting dataset.")
    if eval_dataset is not None and len(eval_dataset) == 0:
        raise ValueError("No usable samples after formatting evaluation dataset.")

    if settings.group_by_length:

        def add_sequence_length_batch(
            batch: dict[str, list[Any]], tokenizer: Any = tokenizer
        ) -> dict[str, list[int]]:
            encoded = tokenizer(
                batch[formatted_text_column],
                add_special_tokens=True,
                truncation=True,
                max_length=settings.max_seq_length,
                padding=False,
            )
            return {
                settings.length_column_name: [
                    len(input_ids) for input_ids in encoded["input_ids"]
                ]
            }

        def add_sequence_lengths(dataset_to_prepare: Any, description: str) -> Any:
            return dataset_to_prepare.map(
                add_sequence_length_batch,
                batched=True,
                batch_size=settings.preprocessing_batch_size,
                num_proc=settings.dataset_num_proc,
                desc=f"Computing {description} sequence lengths",
            )

        dataset = add_sequence_lengths(dataset, "training")
        if eval_dataset is not None:
            eval_dataset = add_sequence_lengths(eval_dataset, "evaluation")

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
        effective_warmup_steps = max(
            1, int(total_steps_for_warmup * settings.warmup_ratio)
        )
    else:
        effective_warmup_steps = settings.warmup_steps

    eval_interval_steps = None
    eval_run_count = None
    chunked_eval_samples_per_run = None
    if eval_dataset is not None:
        eval_interval_steps = settings.eval_steps or settings.save_steps
        if eval_interval_steps <= 0:
            raise ValueError(
                "eval_steps/save_steps must be greater than 0 when evaluation is enabled."
            )
        if eval_sample_strategy != "full":
            eval_run_count = max(
                1,
                total_steps_for_warmup // eval_interval_steps,
            )
            if settings.eval_samples_per_run is not None:
                chunked_eval_samples_per_run = settings.eval_samples_per_run
            elif settings.eval_subset_percent is not None:
                chunked_eval_samples_per_run = max(
                    1,
                    math.ceil(len(eval_dataset) * settings.eval_subset_percent / 100),
                )
            else:
                chunked_eval_samples_per_run = max(
                    1,
                    math.ceil(len(eval_dataset) / eval_run_count),
                )

    training_args_kwargs = {
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
        "bf16": effective_bf16,
        "fp16": effective_fp16,
        "report_to": report_to,
        "run_name": settings.wandb_run_name,
    }
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    training_args_params = set(training_args_signature.parameters.keys())
    if "dataloader_num_workers" in training_args_params:
        training_args_kwargs["dataloader_num_workers"] = settings.dataloader_num_workers
    if "dataloader_pin_memory" in training_args_params:
        training_args_kwargs["dataloader_pin_memory"] = settings.dataloader_pin_memory
    if "dataloader_persistent_workers" in training_args_params:
        training_args_kwargs["dataloader_persistent_workers"] = (
            settings.dataloader_num_workers > 0
        )
    if settings.group_by_length:
        if "group_by_length" in training_args_params:
            training_args_kwargs["group_by_length"] = True
        elif "train_sampling_strategy" in training_args_params:
            training_args_kwargs["train_sampling_strategy"] = "group_by_length"
        else:
            raise RuntimeError(
                "group_by_length is enabled, but this Transformers version does not support length-grouped sampling."
            )
        if "length_column_name" in training_args_params:
            training_args_kwargs["length_column_name"] = settings.length_column_name
    if eval_dataset is not None:
        if "per_device_eval_batch_size" in training_args_params:
            training_args_kwargs["per_device_eval_batch_size"] = settings.eval_batch_size
        if (
            "eval_accumulation_steps" in training_args_params
            and settings.eval_accumulation_steps is not None
        ):
            training_args_kwargs["eval_accumulation_steps"] = settings.eval_accumulation_steps
        if "eval_strategy" in training_args_params:
            training_args_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in training_args_params:
            training_args_kwargs["evaluation_strategy"] = "steps"
        else:
            raise RuntimeError(
                "Evaluation split is enabled, but this Transformers version does not support evaluation_strategy/eval_strategy."
            )
        if "eval_steps" in training_args_params:
            training_args_kwargs["eval_steps"] = eval_interval_steps

    training_args = TrainingArguments(**training_args_kwargs)

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
                "bf16": effective_bf16,
                "fp16": effective_fp16,
            }
            if "dataloader_num_workers" in sft_params:
                sft_kwargs["dataloader_num_workers"] = settings.dataloader_num_workers
            if "dataloader_pin_memory" in sft_params:
                sft_kwargs["dataloader_pin_memory"] = settings.dataloader_pin_memory
            if "dataloader_persistent_workers" in sft_params:
                sft_kwargs["dataloader_persistent_workers"] = (
                    settings.dataloader_num_workers > 0
                )
            if "dataset_num_proc" in sft_params and settings.dataset_num_proc is not None:
                sft_kwargs["dataset_num_proc"] = settings.dataset_num_proc
            if settings.group_by_length:
                if "group_by_length" in sft_params:
                    sft_kwargs["group_by_length"] = True
                elif "train_sampling_strategy" in sft_params:
                    sft_kwargs["train_sampling_strategy"] = "group_by_length"
                else:
                    raise RuntimeError(
                        "group_by_length is enabled, but this TRL SFTConfig version does not support length-grouped sampling."
                    )
                if "length_column_name" in sft_params:
                    sft_kwargs["length_column_name"] = settings.length_column_name
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
            if "padding_free" in sft_params and settings.padding_free is not None:
                sft_kwargs["padding_free"] = settings.padding_free
            if eval_dataset is not None:
                if "per_device_eval_batch_size" in sft_params:
                    sft_kwargs["per_device_eval_batch_size"] = settings.eval_batch_size
                if (
                    "eval_accumulation_steps" in sft_params
                    and settings.eval_accumulation_steps is not None
                ):
                    sft_kwargs["eval_accumulation_steps"] = settings.eval_accumulation_steps
                if "eval_strategy" in sft_params:
                    sft_kwargs["eval_strategy"] = "steps"
                elif "evaluation_strategy" in sft_params:
                    sft_kwargs["evaluation_strategy"] = "steps"
                else:
                    raise RuntimeError(
                        "Evaluation split is enabled, but this TRL SFTConfig version does not support evaluation_strategy/eval_strategy."
                    )
                if "eval_steps" in sft_params:
                    sft_kwargs["eval_steps"] = eval_interval_steps
            args_for_trainer = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": args_for_trainer,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

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
    if "padding_free" in param_names and settings.padding_free is not None:
        trainer_kwargs["padding_free"] = settings.padding_free
    trainer_class = SFTTrainer
    if eval_dataset is not None and chunked_eval_samples_per_run is not None:

        class HereticSubsetEvalSFTTrainer(SFTTrainer):
            _heretic_eval_run_index = 0

            def get_eval_dataloader(self, eval_dataset=None):
                source_dataset = (
                    eval_dataset if eval_dataset is not None else self.eval_dataset
                )
                if (
                    source_dataset is not None
                    and len(source_dataset) > chunked_eval_samples_per_run
                ):
                    if eval_sample_strategy == "chunked":
                        run_index = self._heretic_eval_run_index
                        self._heretic_eval_run_index += 1
                        start = (run_index * chunked_eval_samples_per_run) % len(
                            source_dataset
                        )
                    else:
                        start = 0
                    end = start + chunked_eval_samples_per_run
                    if end <= len(source_dataset):
                        indices = range(start, end)
                    else:
                        indices = list(range(start, len(source_dataset)))
                        indices.extend(range(0, end % len(source_dataset)))

                    if hasattr(source_dataset, "select"):
                        source_dataset = source_dataset.select(indices)
                    else:
                        source_dataset = torch.utils.data.Subset(
                            source_dataset,
                            list(indices),
                        )

                    if hasattr(self, "_eval_dataloaders"):
                        self._eval_dataloaders.pop("eval", None)

                return super().get_eval_dataloader(source_dataset)

        trainer_class = HereticSubsetEvalSFTTrainer

    trainer = trainer_class(**trainer_kwargs)

    if settings.empty_cache_before_save_eval:
        from transformers import TrainerCallback

        class CudaCacheBeforeSaveEvalCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if not torch.cuda.is_available():
                    return control
                if control.should_save or control.should_evaluate:
                    torch.cuda.empty_cache()
                    if settings.memory_debug_prints:
                        print(
                            f"* CUDA cache cleared before save/eval at step {state.global_step}"
                        )
                return control

        trainer.add_callback(CudaCacheBeforeSaveEvalCallback())

    if settings.memory_management_enabled:
        if settings.memory_defrag_every_n_steps <= 0:
            raise ValueError("memory_defrag_every_n_steps must be greater than 0.")

        from transformers import TrainerCallback

        class CudaMemoryDefragCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if not torch.cuda.is_available() or state.global_step <= 0:
                    return control
                if state.global_step % settings.memory_defrag_every_n_steps != 0:
                    return control

                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                slack_gb = (reserved - allocated) / (1024**3)
                reserved_fraction = reserved / total
                should_defrag = (
                    slack_gb >= settings.memory_defrag_reserved_minus_alloc_gb
                    or reserved_fraction >= settings.memory_defrag_target_fraction
                )

                if should_defrag:
                    torch.cuda.empty_cache()

                if settings.memory_debug_prints:
                    action = "empty_cache" if should_defrag else "skip"
                    print(
                        f"* CUDA memory step {state.global_step}: allocated={allocated / (1024**3):.2f}GB, reserved={reserved / (1024**3):.2f}GB, action={action}"
                    )

                return control

        trainer.add_callback(CudaMemoryDefragCallback())

    if settings.nvfp4_emulation:
        from transformers import TrainerCallback

        class Nvfp4ConstraintCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step <= 0:
                    return control
                _constrain_weights_to_nvfp4(model, settings.nvfp4_group_size)
                return control

        trainer.add_callback(Nvfp4ConstraintCallback())

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
    if eval_dataset is not None:
        print(
            f"* Dataset split: train={len(dataset)}, eval={len(eval_dataset)} ({settings.eval_split_percent:.2f}%)"
        )
        if chunked_eval_samples_per_run is not None:
            print(
                f"* Evaluation: every {eval_interval_steps} train steps, "
                f"strategy={eval_sample_strategy}, "
                f"{eval_run_count} scheduled eval runs, "
                f"~{chunked_eval_samples_per_run} eval samples/run"
            )
    precision_desc = "bf16" if effective_bf16 else "fp16" if effective_fp16 else "fp32"
    print(
        f"* Speed config: precision={precision_desc}, tf32={settings.allow_tf32 and torch.cuda.is_available()}, preprocessing_batch_size={settings.preprocessing_batch_size}, dataset_num_proc={settings.dataset_num_proc or 1}"
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
                ("<|turn>user\n", "<|turn>model\n"),
                ("<|turn>user", "<|turn>model"),
                ("<start_of_turn>user\n", "<start_of_turn>model\n"),
                ("<start_of_turn>user", "<start_of_turn>model"),
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

        def template_mask_parts() -> tuple[str | None, str | None]:
            template_name = settings.chat_template_name.lower().replace("_", "-")
            if template_name in {"gemma-4", "gemma4", "gemma-4-thinking"}:
                return "<|turn>user\n", "<|turn>model\n"
            if template_name in {
                "gemma",
                "gemma-2",
                "gemma2",
                "gemma-3",
                "gemma3",
                "gemma-3n",
                "gemma3n",
            }:
                return "<start_of_turn>user\n", "<start_of_turn>model\n"
            if template_name in {"llama-3", "llama3", "llama-3.1", "llama-3.2"}:
                return (
                    "<|start_header_id|>user<|end_header_id|>\n\n",
                    "<|start_header_id|>assistant<|end_header_id|>\n\n",
                )
            if template_name in {"chatml", "gemma-chatml"}:
                return "<|im_start|>user\n", "<|im_start|>assistant\n"
            return None, None

        instruction_part = settings.instruction_part
        response_part = settings.response_part

        if instruction_part is None or response_part is None:
            inferred_instruction, inferred_response = infer_mask_parts(sample_text)

            if inferred_instruction is None or inferred_response is None:
                template_instruction, template_response = template_mask_parts()
                if inferred_instruction is None:
                    inferred_instruction = template_instruction
                if inferred_response is None:
                    inferred_response = template_response

            if instruction_part is None:
                instruction_part = inferred_instruction
            if response_part is None:
                response_part = inferred_response

            if instruction_part is None or response_part is None:
                raise ValueError(
                    "Could not infer response-only masking delimiters. Set instruction_part and response_part explicitly in the stage config."
                )

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
        # Save tokenizer to checkpoint directory when interrupted
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        try:
            tokenizer.save_pretrained(str(checkpoint_path))
            print(f"* Saved tokenizer to checkpoint directory: {checkpoint_path}")
        except Exception as e:
            print(
                f"[yellow]* Warning: could not save tokenizer to checkpoint directory: {e}[/]"
            )

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer to checkpoint directory after training completes
    try:
        tokenizer.save_pretrained(str(checkpoint_path))
        print(f"* Saved tokenizer to checkpoint directory: {checkpoint_path}")
    except Exception as e:
        print(
            f"[yellow]* Warning: could not save tokenizer to checkpoint directory: {e}[/]"
        )

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
                export_model_snapshot(model, tokenizer)
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
