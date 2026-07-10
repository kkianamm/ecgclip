"""BiomedCoOp-style prompt learning adapted to multi-label PTB-XL.

Key differences from the original image-classification implementation:
1. Uses standard pip `open-clip-torch` rather than the repository's modified
   encode_text API.
2. Feeds learnable prompt embeddings through the Hugging Face text tower via
   `inputs_embeds`.
3. Uses independent multi-label logits, BCE supervision, and Bernoulli
   knowledge distillation instead of softmax cross-entropy.
4. Accepts cached, L2-normalized BiomedCLIP ECG image embeddings, so only the
   prompt context vectors are trained.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


class LearnableECGPrompt(nn.Module):
    """CoOp context tokens inserted after the BERT [CLS] token."""

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        class_names: Sequence[str],
        class_texts: Mapping[str, str],
        n_ctx: int = 4,
        ctx_init: Optional[str] = "an ECG pattern showing",
        class_specific: bool = False,
        context_length: int = 256,
    ) -> None:
        super().__init__()

        if not hasattr(clip_model, "text"):
            raise TypeError(
                "Expected BiomedCLIP to expose `model.text` as an HFTextEncoder."
            )

        text_tower = clip_model.text
        if not hasattr(text_tower, "transformer"):
            raise TypeError("BiomedCLIP text tower has no Hugging Face transformer.")

        self.class_names = list(class_names)
        self.n_cls = len(self.class_names)
        self.n_ctx = int(n_ctx)
        self.class_specific = bool(class_specific)

        placeholder = " ".join(["X"] * self.n_ctx)
        prompt_strings = [
            f"{placeholder} {class_texts[name]}."
            for name in self.class_names
        ]

        tokenized = tokenizer(prompt_strings, context_length=context_length)
        embedding_layer = text_tower.transformer.get_input_embeddings()
        embedding_device = embedding_layer.weight.device
        tokenized = tokenized.to(embedding_device)

        with torch.no_grad():
            base_embeddings = embedding_layer(tokenized).detach().float()

        hidden_dim = int(base_embeddings.shape[-1])
        if ctx_init:
            init_tokens = tokenizer([ctx_init], context_length=context_length).to(
                embedding_device
            )
            with torch.no_grad():
                init_embeddings = embedding_layer(init_tokens)[0, 1 : 1 + self.n_ctx]
            if init_embeddings.shape[0] != self.n_ctx:
                raise ValueError(
                    f"ctx_init did not provide {self.n_ctx} context tokens."
                )
            ctx = init_embeddings.detach().clone().float()
        else:
            ctx = torch.empty(self.n_ctx, hidden_dim, device=embedding_device)
            nn.init.normal_(ctx, std=0.02)

        if self.class_specific:
            ctx = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)

        self.ctx = nn.Parameter(ctx)
        pad_id = int(text_tower.config.pad_token_id)
        attention_mask = (tokenized != pad_id).long()

        self.register_buffer("tokenized_prompts", tokenized, persistent=True)
        self.register_buffer("base_embeddings", base_embeddings, persistent=False)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def prompt_embeddings(self) -> torch.Tensor:
        """Construct [class, sequence, hidden] prompt embeddings."""
        ctx = self.ctx.float()
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Position 0 is BERT [CLS]. The next n_ctx placeholder embeddings are
        # replaced by trainable context vectors.
        return torch.cat(
            [
                self.base_embeddings[:, :1, :],
                ctx,
                self.base_embeddings[:, 1 + self.n_ctx :, :],
            ],
            dim=1,
        )

    def encode(self, text_tower: nn.Module) -> torch.Tensor:
        """Encode continuous prompts using standard HF `inputs_embeds`."""
        prompt_embeddings = self.prompt_embeddings()
        outputs = text_tower.transformer(
            inputs_embeds=prompt_embeddings,
            attention_mask=self.attention_mask,
        )
        pooled = text_tower.pooler(outputs, self.attention_mask)
        projected = text_tower.proj(pooled)
        return projected


class ECGBiomedCoOp(nn.Module):
    """Frozen BiomedCLIP plus trainable ECG prompt context vectors."""

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        class_names: Sequence[str],
        class_texts: Mapping[str, str],
        teacher_prompt_bank: Mapping[str, Sequence[str]],
        n_ctx: int = 4,
        ctx_init: Optional[str] = "an ECG pattern showing",
        class_specific: bool = False,
        context_length: int = 256,
        teacher_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.clip = clip_model
        self.class_names = list(class_names)

        # Freeze the complete foundation model.
        self.clip.eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        self.prompt_learner = LearnableECGPrompt(
            clip_model=self.clip,
            tokenizer=tokenizer,
            class_names=self.class_names,
            class_texts=class_texts,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            class_specific=class_specific,
            context_length=context_length,
        )

        teacher_features = self._build_teacher_features(
            tokenizer=tokenizer,
            prompt_bank=teacher_prompt_bank,
            context_length=context_length,
            batch_size=teacher_batch_size,
        )
        semantic_target = _l2_normalize(teacher_features.mean(dim=1))

        self.register_buffer("teacher_features", teacher_features, persistent=False)
        self.register_buffer("semantic_target", semantic_target, persistent=False)

    def train(self, mode: bool = True):
        # Keep dropout disabled in the frozen image/text towers.
        super().train(mode)
        self.clip.eval()
        self.prompt_learner.train(mode)
        return self

    @torch.no_grad()
    def _build_teacher_features(
        self,
        tokenizer,
        prompt_bank: Mapping[str, Sequence[str]],
        context_length: int,
        batch_size: int,
    ) -> torch.Tensor:
        counts = [len(prompt_bank[name]) for name in self.class_names]
        if min(counts) == 0:
            raise ValueError("Every class must have at least one teacher prompt.")
        if len(set(counts)) != 1:
            raise ValueError(
                "All classes must have the same number of teacher prompts; "
                f"received counts={counts}."
            )

        n_prompts = counts[0]
        flat_prompts: List[str] = [
            prompt_bank[class_name][prompt_idx]
            for class_name in self.class_names
            for prompt_idx in range(n_prompts)
        ]

        device = next(self.clip.parameters()).device
        features: List[torch.Tensor] = []
        for start in range(0, len(flat_prompts), batch_size):
            batch = flat_prompts[start : start + batch_size]
            tokens = tokenizer(batch, context_length=context_length).to(device)
            encoded = self.clip.encode_text(tokens)
            features.append(_l2_normalize(encoded).cpu())

        stacked = torch.cat(features, dim=0)
        stacked = stacked.reshape(len(self.class_names), n_prompts, -1)
        return stacked.to(device)

    @torch.no_grad()
    def select_teacher_features(
        self,
        image_features: torch.Tensor,
        tau: float = 1.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """BiomedCoOp statistics-based prompt pruning.

        A prompt index is scored by its mean, over the batch, of the strongest
        class similarity. Robust median/MAD z-scores remove prompt outliers.
        """
        image_features = _l2_normalize(image_features)
        teacher = self.teacher_features.float()  # [classes, prompts, dim]
        scale = self.clip.logit_scale.exp().detach().float()

        # [batch, prompts, classes]
        prompt_logits = scale * torch.einsum(
            "bd,cpd->bpc", image_features, teacher
        )
        prompt_scores = prompt_logits.max(dim=-1).values.mean(dim=0)

        median = prompt_scores.median()
        mad = (prompt_scores - median).abs().median().clamp_min(1e-6)
        robust_z = 0.6745 * (prompt_scores - median) / mad
        mask = robust_z.abs() <= float(tau)

        # Keep training numerically safe for a very homogeneous or tiny batch.
        if not torch.any(mask):
            keep = max(1, teacher.shape[1] // 2)
            best = robust_z.abs().argsort()[:keep]
            mask = torch.zeros_like(robust_z, dtype=torch.bool)
            mask[best] = True

        selected = _l2_normalize(teacher[:, mask, :].mean(dim=1))
        return selected, mask, prompt_scores

    def learned_text_features(self) -> torch.Tensor:
        encoded = self.prompt_learner.encode(self.clip.text)
        return _l2_normalize(encoded)

    def student_logits(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = _l2_normalize(image_features)
        text_features = self.learned_text_features()
        scale = self.clip.logit_scale.exp().detach().float()
        return scale * image_features @ text_features.t()

    def forward_features(
        self,
        image_features: torch.Tensor,
        tau: float = 1.5,
    ) -> Dict[str, torch.Tensor]:
        image_features = _l2_normalize(image_features)
        text_features = self.learned_text_features()
        scale = self.clip.logit_scale.exp().detach().float()

        student_logits = scale * image_features @ text_features.t()

        selected_teacher, mask, prompt_scores = self.select_teacher_features(
            image_features, tau=tau
        )
        teacher_logits = scale * image_features @ selected_teacher.t()

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits.detach(),
            "text_features": text_features,
            "semantic_target": self.semantic_target.detach(),
            "selected_prompt_mask": mask,
            "prompt_scores": prompt_scores,
        }


def bernoulli_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Knowledge distillation for independent multi-label Bernoulli outputs."""
    temperature = float(temperature)
    teacher_prob = torch.sigmoid(teacher_logits / temperature).detach()
    loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        teacher_prob,
    )
    return loss * (temperature ** 2)
