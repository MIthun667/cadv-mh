from __future__ import annotations

import time
import warnings
from collections import defaultdict
from typing import Any

import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, objective, optimizer, device, gradient_accumulation=1, gradient_clip=1.0, fp16=True, fusion_mode="full"):
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.gradient_accumulation = gradient_accumulation
        self.gradient_clip = gradient_clip
        self.amp_enabled = bool(fp16 and self.device.type == "cuda")
        self.fusion_mode = fusion_mode
        # RTX 20-series cards can overflow at GradScaler's large default scale
        # during the first pretrained-transformer update. A conservative scale
        # still retains FP16's useful dynamic-range protection.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0, growth_interval=2000)

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def train_epoch(self, loader, stage: int, temperature: float, mask_token_id: int, max_batches: int | None = None) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        predictions, targets = [], []
        skipped_amp_steps = 0
        consecutive_amp_overflows = 0
        processed_batches = 0
        started = time.perf_counter()
        total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
        progress = tqdm(loader, total=total_batches, desc=f"Stage {stage} train", unit="batch", dynamic_ncols=True)
        for step, raw in enumerate(progress, 1):
            if max_batches is not None and step > max_batches:
                break
            processed_batches += 1
            batch = self._move(raw)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                output = self.model(
                    batch["input_ids"], batch["attention_mask"], batch["special_token_mask"], mask_token_id,
                    stage=stage, temperature=temperature, enable_evidence=stage > 1,
                    fusion_mode=self.fusion_mode,
                )
                augmented_logits = None
                if stage == 3 and self.objective.weights.consistency and "augmented_input_ids" in batch:
                    augmented_logits = self.model(
                        batch["augmented_input_ids"], batch["augmented_attention_mask"], batch["augmented_special_token_mask"],
                        mask_token_id, stage=1, return_auxiliary=False,
                    ).original_logits
                losses = self.objective(output, batch["labels"], augmented_logits=augmented_logits, stage=stage)
                loss = losses["total"] / self.gradient_accumulation
            self.scaler.scale(loss).backward()
            final_requested_batch = max_batches is not None and step == min(max_batches, len(loader))
            if step % self.gradient_accumulation == 0 or step == len(loader) or final_requested_batch:
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                if not torch.isfinite(norm):
                    if not self.amp_enabled:
                        raise FloatingPointError("Non-finite gradient norm without AMP")
                    skipped_amp_steps += 1
                    consecutive_amp_overflows += 1
                    old_scale = self.scaler.get_scale()
                    # GradScaler records the non-finite result during unscale_,
                    # skips optimizer.step, and lowers its scale in update().
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    warnings.warn(
                        f"AMP overflow at batch {step}; optimizer step skipped and loss scale reduced "
                        f"from {old_scale:g} to {self.scaler.get_scale():g}",
                        RuntimeWarning,
                    )
                    if consecutive_amp_overflows >= 8:
                        raise FloatingPointError("Eight consecutive AMP overflows; aborting instead of silently skipping training")
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    consecutive_amp_overflows = 0
                self.optimizer.zero_grad(set_to_none=True)
            for key, value in losses.items():
                totals[key] += float(value.detach())
            predictions.extend(output.final_logits.argmax(-1).detach().cpu().tolist())
            targets.extend(batch["labels"].detach().cpu().tolist())
            if step == 1 or step % 25 == 0 or step == total_batches:
                progress.set_postfix(
                    loss=f"{float(losses['total'].detach()):.4f}",
                    amp_scale=f"{self.scaler.get_scale():.0f}" if self.amp_enabled else "off",
                    skipped=skipped_amp_steps,
                )
        count = max(1, processed_batches)
        result = {f"train_{key}": value / count for key, value in totals.items()}
        result["train_macro_f1"] = f1_score(targets, predictions, average="macro", zero_division=0)
        result["runtime_seconds"] = time.perf_counter() - started
        result["skipped_amp_steps"] = skipped_amp_steps
        result["final_amp_scale"] = float(self.scaler.get_scale()) if self.amp_enabled else 1.0
        if self.device.type == "cuda":
            result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
        return result

    @torch.no_grad()
    def evaluate(self, loader, stage: int, mask_token_id: int, max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        predictions, targets = [], []
        losses_total = 0.0
        processed_batches = 0
        total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
        progress = tqdm(loader, total=total_batches, desc=f"Stage {stage} validation", unit="batch", dynamic_ncols=True)
        for step, raw in enumerate(progress, 1):
            if max_batches is not None and step > max_batches:
                break
            processed_batches += 1
            batch = self._move(raw)
            output = self.model(
                batch["input_ids"], batch["attention_mask"], batch["special_token_mask"], mask_token_id,
                stage=stage, selection_mode="top_rho", enable_evidence=stage > 1,
                fusion_mode=self.fusion_mode,
            )
            augmented_logits = None
            if stage == 3 and self.objective.weights.consistency and "augmented_input_ids" in batch:
                augmented_logits = self.model(
                    batch["augmented_input_ids"], batch["augmented_attention_mask"], batch["augmented_special_token_mask"],
                    mask_token_id, stage=1, return_auxiliary=False,
                ).original_logits
            losses = self.objective(output, batch["labels"], augmented_logits=augmented_logits, stage=stage)
            losses_total += float(losses["total"])
            predictions.extend(output.final_logits.argmax(-1).cpu().tolist())
            targets.extend(batch["labels"].cpu().tolist())
            if step == 1 or step % 25 == 0 or step == total_batches:
                progress.set_postfix(loss=f"{float(losses['total']):.4f}")
        return {
            "validation_loss": losses_total / max(1, processed_batches),
            "validation_macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        }
