"""Training-only veRL drivers, without benchmark evaluation or seed selection."""

import math
import time

import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


class TrainingOnlyPPOTrainer(RayPPOTrainer):
    """Use the native GRPO loop and checkpoints without a validation dataset."""

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        if train_dataset is None:
            raise ValueError("A training dataset is required.")
        self.train_dataset, self.val_dataset = train_dataset, None
        generator = torch.Generator().manual_seed(self.config.data.seed)
        sampler = train_sampler or RandomSampler(train_dataset, generator=generator)
        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=sampler,
            num_workers=self.config.data.dataloader_num_workers,
            collate_fn=collate_fn or default_collate_fn,
            drop_last=True,
        )
        self.val_dataloader = None
        if not len(self.train_dataloader):
            raise ValueError("Training data contains fewer examples than data.train_batch_size.")
        self.total_training_steps = (
            self.config.trainer.total_training_steps
            or len(self.train_dataloader) * self.config.trainer.total_epochs
        )
        if self.total_training_steps < 1:
            raise ValueError("Training step limit must be positive.")
        with open_dict(self.config):
            # veRL's scheduler advances once per iteration (not per minibatch).
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                self.total_training_steps
            )
            self.config.trainer.total_epochs = max(
                self.config.trainer.total_epochs,
                math.ceil(self.total_training_steps / len(self.train_dataloader)) + 1,
            )


class DistillationTrainer(TrainingOnlyPPOTrainer):
    """Student rollout -> online teacher logits -> full-vocabulary update."""

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        if self.global_steps >= self.total_training_steps:
            return
        fixed = self.config.method in {"kd", "sft"}
        for _ in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                start = time.monotonic()
                batch = DataProto.from_single_dict(batch_dict)
                self.global_steps += 1
                if not fixed:
                    prompts = self._get_gen_batch(batch)
                    prompts = prompts.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    prompts.meta_info["global_steps"] = self.global_steps
                    batch = self.actor_rollout_wg.generate_sequences(prompts)
                    length = batch.batch["responses"].shape[-1]
                    batch.batch["response_mask"] = batch.batch["attention_mask"][:, -length:].bool()
                batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                result = self.actor_rollout_wg.update_actor(batch)
                metrics = reduce_metrics(result.meta_info["metrics"])
                metrics["timing/iteration_seconds"] = time.monotonic() - start
                metrics["training/iteration"] = self.global_steps
                final = self.global_steps >= self.total_training_steps
                freq = self.config.trainer.save_freq
                if final or (freq > 0 and self.global_steps % freq == 0):
                    self._save_checkpoint()
                logger.log(metrics, step=self.global_steps)
                if final:
                    return
