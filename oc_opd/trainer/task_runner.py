"""Construct the real Ray worker groups used by each method."""

import ray
from omegaconf import OmegaConf

from oc_opd.data.dataset import TrainingDataset
from oc_opd.data.prompts import validate_tokenizers
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from .ray_trainer import DistillationTrainer, TrainingOnlyPPOTrainer
from .validation import validate_training_config


class TrainingTaskRunner:
    def run(self, config):
        validate_training_config(config)
        OmegaConf.resolve(config)
        tokenizer = hf_tokenizer(
            copy_to_local(config.actor_rollout_ref.model.path),
            trust_remote_code=config.actor_rollout_ref.model.trust_remote_code,
        )
        if config.actor_rollout_ref.model.custom_chat_template is not None:
            tokenizer.chat_template = config.actor_rollout_ref.model.custom_chat_template
        is_grpo = config.method == "grpo"
        if not is_grpo and config.method != "sft":
            teacher_tokenizer = hf_tokenizer(
                copy_to_local(config.actor_rollout_ref.distillation.teacher_path),
                trust_remote_code=config.actor_rollout_ref.model.trust_remote_code,
            )
            validate_tokenizers(tokenizer, teacher_tokenizer)
        dataset = TrainingDataset(
            config.data.train_files,
            tokenizer,
            config.data,
            fixed_responses=config.method in {"kd", "sft"},
        )
        # Check one example before allocating GPU workers; all rows are also
        # validated as they are read, including prompt lengths and KD responses.
        dataset[0]
        reward = None
        if is_grpo:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker

            worker = ActorRolloutRefWorker
            trainer_cls = TrainingOnlyPPOTrainer
            reward = load_reward_manager(config, tokenizer, num_examine=0)
        else:
            from oc_opd.workers.fsdp_workers import DistillationWorker

            worker = DistillationWorker
            trainer_cls = DistillationTrainer
        roles = {Role.ActorRollout: ray.remote(worker)}
        mapping = {Role.ActorRollout: "global_pool"}
        if is_grpo and (
            config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward
        ):
            roles[Role.RefPolicy] = ray.remote(worker)
            mapping[Role.RefPolicy] = "global_pool"
        pool = ResourcePoolManager(
            resource_pool_spec={
                "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes
            },
            mapping=mapping,
        )
        trainer = trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=None,
            role_worker_mapping=roles,
            resource_pool_manager=pool,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward,
            val_reward_fn=None,
            train_dataset=dataset,
        )
        trainer.init_workers()
        trainer.fit()
