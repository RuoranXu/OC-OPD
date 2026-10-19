"""Distillation actor with distributed response averaging and veRL optimizers."""

import torch
import torch.distributed as dist

from oc_opd.losses.training import response_loss_sum
from verl.utils.device import get_device_id
from verl.workers.actor.dp_actor import DataParallelPPOActor


class DistillationActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer, *, teacher, method_config):
        super().__init__(config, actor_module, actor_optimizer)
        self.teacher = teacher
        self.method_config = method_config

    def update_policy(self, data):
        self.actor_module.train()
        if self.teacher is not None:
            self.teacher.eval()
        device = get_device_id()
        world_size = dist.get_world_size()
        metrics = {"actor/grad_norm": [], "actor/skipped_nonfinite_updates": []}
        aggregate = torch.zeros(3, device=device)
        valid_responses = torch.zeros((), device=device)
        reduction = self.method_config.get("loss_reduction", "sequence_mean")
        mini_size = self.config.ppo_mini_batch_size  # normalized by veRL worker
        micro_size = self.config.ppo_micro_batch_size_per_gpu
        for mini in data.split(mini_size):
            lengths = mini.batch["response_mask"].sum(dim=-1)
            counts = torch.stack((lengths.sum(), (lengths > 0).sum())).to(
                device, dtype=torch.float32
            )
            dist.all_reduce(counts)
            if counts[0].item() == 0:
                raise ValueError("The entire distributed minibatch has no response tokens.")
            denominator = counts[1] if reduction == "sequence_mean" else counts[0]
            valid_responses += counts[1]
            self.actor_optimizer.zero_grad(set_to_none=True)
            for micro in mini.split(micro_size):
                micro = micro.to(device)
                batch = micro.batch
                length = batch["responses"].shape[-1]
                inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids")}
                teacher_logits = None
                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    if self.teacher is not None:
                        with torch.no_grad():
                            teacher_logits = self.teacher(**inputs, use_cache=False).logits
                            teacher_logits = teacher_logits[:, -length - 1 : -1]
                    student_logits = self.actor_module(**inputs, use_cache=False).logits
                    student_logits = student_logits[:, -length - 1 : -1]
                loss, stats = response_loss_sum(
                    student_logits,
                    teacher_logits,
                    batch["responses"],
                    batch["response_mask"],
                    method=self.method_config.name,
                    chunk_size=self.method_config.logit_chunk_size,
                    controller=dict(self.method_config.controller),
                    reduction=reduction,
                )
                # FSDP averages rank gradients. The loss sums per-response
                # token means; normalize by the GLOBAL nonempty-response count.
                # token_sum retains the explicit global-token-mean alternative.
                loss = loss * world_size / denominator
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                aggregate += stats
                del student_logits, teacher_logits, loss
            norm = self._optimizer_step()
            if not torch.isfinite(norm):
                raise FloatingPointError(
                    "Non-finite actor gradients: veRL skipped the optimizer step; "
                    "training stopped rather than consuming the update budget."
                )
            metrics["actor/grad_norm"].append(float(norm.detach()))
            metrics["actor/skipped_nonfinite_updates"].append(0.0)
        self.actor_optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(aggregate)
        count = aggregate[2].clamp_min(1)
        if self.teacher is not None:
            metrics["distill/reverse_kl"] = float(aggregate[0] / count)
            metrics["distill/forward_kl"] = float(aggregate[1] / count)
        metrics["distill/response_tokens"] = float(aggregate[2])
        metrics["distill/valid_responses"] = float(valid_responses)
        metrics["actor/updates_per_iteration"] = len(metrics["actor/grad_norm"])
        return metrics
