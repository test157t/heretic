# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Settings
from .model import Model
from .utils import Prompt, load_prompts, print


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_top5: Tensor
    base_top10: Tensor
    base_refusals: int

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)
        self.base_top5 = self.base_logprobs.topk(5, dim=1).indices
        self.base_top10 = self.base_logprobs.topk(10, dim=1).indices

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self) -> int:
        refusal_count = 0

        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        for prompt, response in zip(self.bad_prompts, responses):
            is_refusal = self.is_refusal(response)
            if is_refusal:
                refusal_count += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{response}[/]"
                )

        if self.settings.print_responses:
            print()

        return refusal_count

    def get_score(self) -> tuple[tuple[float, float], float, float, float, float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)

        top5 = logprobs.topk(5, dim=1).indices
        top10 = logprobs.topk(10, dim=1).indices

        top5_ordered = (top5 == self.base_top5).all(dim=1).float().mean().item()
        top10_unordered = (
            top10.sort(dim=1).values == self.base_top10.sort(dim=1).values
        ).all(dim=1).float().mean().item()

        print(f"  * Top 5 ordered: [bold]{top5_ordered:.2%}[/]")
        print(f"  * Top 10 unordered: [bold]{top10_unordered:.2%}[/]")

        # Mean Hellinger distance between first-token distributions.
        # H(P, Q) = sqrt(1 - sum_i sqrt(P_i * Q_i)).
        # We compute this in log-space for numerical stability.
        bhattacharyya = torch.exp(0.5 * (logprobs + self.base_logprobs)).sum(dim=1)
        hellinger_distance = torch.sqrt((1.0 - bhattacharyya).clamp(min=0.0)).mean().item()
        print(f"  * Hellinger distance: [bold]{hellinger_distance:.4f}[/]")

        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        refusals_score = refusals / self.base_refusals

        if kl_divergence >= kl_divergence_target:
            kld_score = kl_divergence / kl_divergence_scale
        else:
            kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

        score = (
            kld_score,
            refusals_score,
        )

        return (
            score,
            kl_divergence,
            hellinger_distance,
            top5_ordered,
            top10_unordered,
            refusals,
        )
