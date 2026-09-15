# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import sys
import warnings
import functools


# This is emitted by mergekit when imported through the Unsloth stack. It is a
# dependency compatibility warning and does not affect Heretic's runtime behavior.
warnings.filterwarnings(
    "ignore",
    message=r".*MultislerpMergeTask.*shadows an attribute in parent.*",
    category=UserWarning,
)


# Transformers 5.x moved AttentionMaskConverter from modeling_attn_mask_utils to
# masking_utils, but Unsloth still imports the deprecated API. The functions in
# modeling_attn_mask_utils still work — they just emit FutureWarnings that clutter
# output. We strip those warnings here so the deprecated backward-compat shim
# does its job silently until Unsloth migrates.

def _patch_modeling_attn_mask_utils():
    try:
        import transformers.modeling_attn_mask_utils as _mam
    except ImportError:
        return

    import torch

    # --- Patch AttentionMaskConverter methods ---

    # __init__ without the warning
    _original_init = _mam.AttentionMaskConverter.__init__

    @functools.wraps(_original_init)
    def _patched_init(self, is_causal, sliding_window=None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is "
                f"a strictly positive integer, not `{self.sliding_window}`"
            )

    _mam.AttentionMaskConverter.__init__ = _patched_init

    # _make_causal_mask without the warning
    @staticmethod
    @functools.wraps(_mam.AttentionMaskConverter._make_causal_mask)
    def _patched_make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0, sliding_window=None):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
            )
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1
            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            from transformers.utils.import_utils import is_torchdynamo_compiling
            if is_torchdynamo_compiling():
                mask = mask.clone()
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    _mam.AttentionMaskConverter._make_causal_mask = _patched_make_causal_mask

    # _expand_mask without the warning
    @staticmethod
    @functools.wraps(_mam.AttentionMaskConverter._expand_mask)
    def _patched_expand_mask(mask, dtype, tgt_len=None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    _mam.AttentionMaskConverter._expand_mask = _patched_expand_mask

    # _unmask_unattended without the warning
    @staticmethod
    @functools.wraps(_mam.AttentionMaskConverter._unmask_unattended)
    def _patched_unmask_unattended(expanded_mask, min_dtype):
        if expanded_mask.dtype == torch.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )
        return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True))

    _mam.AttentionMaskConverter._unmask_unattended = _patched_unmask_unattended

    # _ignore_causal_mask_sdpa without the warning
    @staticmethod
    @functools.wraps(_mam.AttentionMaskConverter._ignore_causal_mask_sdpa)
    def _patched_ignore_causal_mask_sdpa(attention_mask, inputs_embeds, past_key_values_length, sliding_window=None, is_training=False):
        from transformers.utils.import_utils import is_tracing
        _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        key_value_length = query_length + past_key_values_length
        is_tracing_ = is_tracing(inputs_embeds)
        ignore_causal_mask = False
        if attention_mask is None:
            if (
                (is_training or not is_tracing_)
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(attention_mask.shape) == 4:
                return False
            elif not is_tracing_ and torch.all(attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    ignore_causal_mask = True
        return ignore_causal_mask

    _mam.AttentionMaskConverter._ignore_causal_mask_sdpa = _patched_ignore_causal_mask_sdpa

    # Module-level _prepare_4d_attention_mask_for_sdpa without the warning
    def _patched_prepare_4d_attention_mask_for_sdpa(mask, dtype, tgt_len=None):
        from transformers.utils.import_utils import is_tracing
        _, key_value_length = mask.shape
        tgt_len = tgt_len if tgt_len is not None else key_value_length
        if not is_tracing(mask) and torch.all(mask == 1):
            return None
        else:
            return _mam.AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)

    _mam._prepare_4d_attention_mask_for_sdpa = _patched_prepare_4d_attention_mask_for_sdpa


_patch_modeling_attn_mask_utils()
