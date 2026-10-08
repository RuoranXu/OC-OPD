"""Fail early for runtime features not implemented by the distillation adapter."""


def validate_training_config(config):
    method = config.method
    if method not in {"oc_opd", "opd", "fkl", "kd", "sft", "grpo"}:
        raise ValueError(f"Unknown training method: {method}")
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    model = config.actor_rollout_ref.model
    if not config.data.train_files or not model.path:
        raise ValueError("Specify data.train_files and actor_rollout_ref.model.path.")
    if rollout.mode != "sync":
        raise ValueError("These recipes use synchronous veRL rollout.")
    if config.data.prompt_mode not in {"base", "chat"}:
        raise ValueError("data.prompt_mode must be base or chat.")
    world = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if (
        world < 1
        or actor.ppo_mini_batch_size < 1
        or config.data.train_batch_size < 1
        or config.data.train_batch_size % actor.ppo_mini_batch_size
    ):
        raise ValueError("Training batch must be divisible by PPO minibatch size.")
    mini = actor.ppo_mini_batch_size * rollout.n
    micro = actor.ppo_micro_batch_size_per_gpu
    if micro is None or micro < 1 or mini % (world * micro):
        raise ValueError("Minibatch * rollout.n must divide evenly across GPUs and microbatches.")
    if actor.strategy != "fsdp":
        raise ValueError("The supplied worker recipes use FSDP1.")
    if method == "grpo":
        if rollout.n < 2 or config.algorithm.adv_estimator != "grpo":
            raise ValueError("GRPO requires group size >= 2 and GRPO advantages.")
        return
    if actor.ulysses_sequence_parallel_size != 1:
        raise ValueError(
            "Full-vocabulary adapter currently supports FSDP1 and sequence parallel=1."
        )
    if model.use_remove_padding or model.use_fused_kernels or model.get("use_liger", False):
        raise ValueError("Distillation requires padded, unfused full-vocabulary logits.")
    if model.lora_rank or model.lora_adapter_path:
        raise ValueError("Distillation recipes currently train full student parameters.")
    if actor.use_dynamic_bsz or actor.ppo_epochs != 1 or actor.shuffle:
        raise ValueError("Distillation uses fixed microbatches, one pass, and no inner shuffling.")
    if actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        raise ValueError("Distillation objectives replace, rather than add to, PPO/reference KL.")
    if method != "sft" and not config.actor_rollout_ref.distillation.teacher_path:
        raise ValueError("Specify actor_rollout_ref.distillation.teacher_path.")
    if method in {"kd", "sft"} and rollout.n != 1:
        raise ValueError("Fixed-response KD/SFT requires rollout.n=1.")
