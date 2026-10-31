"""Fail early for runtime features not implemented by the distillation adapter."""

from omegaconf import open_dict

from oc_opd.config import OCOPDConfig


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def configure_finetuning(config):
    """Resolve the public full/LoRA switch before veRL creates model workers."""
    kind = config.get("finetuning_type", "full")
    if kind not in {"full", "lora"}:
        raise ValueError("finetuning_type must be full or lora.")
    group = config.actor_rollout_ref
    model = group.model
    if kind == "full":
        if (
            model.get("lora_rank", 0)
            or model.get("lora_adapter_path")
            or config.get("lora_adapter_path")
        ):
            raise ValueError("Use finetuning_type=lora to enable student adapters.")
        return

    rank = config.get("lora_rank", 16)
    alpha = config.get("lora_alpha", 32)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 512:
        raise ValueError("lora_rank must be an integer in [1, 512].")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha < 1:
        raise ValueError("lora_alpha must be a positive integer.")
    targets = config.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if isinstance(targets, str) or not targets:
        raise ValueError("lora_target_modules must be an explicit nonempty module-name list.")
    targets = list(targets)
    if any(not isinstance(name, str) or not name for name in targets):
        raise ValueError("LoRA target module names must be nonempty strings.")
    if len(set(targets)) != len(targets) or any(
        name.rsplit(".", 1)[-1] in {"lm_head", "embed_tokens"} for name in targets
    ):
        raise ValueError(
            "LoRA targets must be unique attention/MLP modules, excluding embeddings and lm_head."
        )
    adapter = config.get("lora_adapter_path")
    if model.get("lora_rank", 0) not in {0, rank} or model.get("lora_adapter_path") not in {
        None,
        adapter,
    }:
        raise ValueError("Set LoRA options through the top-level lora_* fields consistently.")
    if group.rollout.name != "vllm":
        raise ValueError("The LoRA recipe uses veRL's vLLM adapter weight synchronization.")
    with open_dict(model):
        model.lora_rank = rank
        model.lora_alpha = alpha
        model.target_modules = targets
        model.lora_adapter_path = adapter
    with open_dict(group.actor.fsdp_config):
        group.actor.fsdp_config.use_orig_params = True
    with open_dict(group.rollout):
        # A dummy base sync alone omits a restored nonzero adapter on the first
        # rollout. Preload base weights, then synchronize adapters every rollout.
        # Layered summon also selects sleep level 1, retaining base weights.
        group.rollout.load_format = "safetensors"
        group.rollout.layered_summon = True


def validate_training_config(config):
    configure_finetuning(config)
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
    world = _positive_integer(config.trainer.n_gpus_per_node, "trainer.n_gpus_per_node")
    world *= _positive_integer(config.trainer.nnodes, "trainer.nnodes")
    _positive_integer(actor.ppo_mini_batch_size, "actor.ppo_mini_batch_size")
    _positive_integer(config.data.train_batch_size, "data.train_batch_size")
    _positive_integer(rollout.n, "rollout.n")
    micro = _positive_integer(
        actor.ppo_micro_batch_size_per_gpu, "actor.ppo_micro_batch_size_per_gpu"
    )
    if config.trainer.get("total_training_steps") is not None:
        _positive_integer(config.trainer.total_training_steps, "trainer.total_training_steps")
    if (
        world < 1
        or actor.ppo_mini_batch_size < 1
        or config.data.train_batch_size < 1
        or config.data.train_batch_size % actor.ppo_mini_batch_size
    ):
        raise ValueError("Training batch must be divisible by PPO minibatch size.")
    mini = actor.ppo_mini_batch_size * rollout.n
    if mini % (world * micro):
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
    reduction = config.actor_rollout_ref.distillation.get("loss_reduction", "sequence_mean")
    if reduction not in {"sequence_mean", "token_sum"}:
        raise ValueError("distillation.loss_reduction must be sequence_mean or token_sum.")
    _positive_integer(
        config.actor_rollout_ref.distillation.logit_chunk_size, "distillation.logit_chunk_size"
    )
    if method != "sft":
        OCOPDConfig(**dict(config.actor_rollout_ref.distillation.controller))
    if actor.use_dynamic_bsz or actor.ppo_epochs != 1 or actor.shuffle:
        raise ValueError("Distillation uses fixed microbatches, one pass, and no inner shuffling.")
    if actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        raise ValueError("Distillation objectives replace, rather than add to, PPO/reference KL.")
    if method != "sft" and not config.actor_rollout_ref.distillation.teacher_path:
        raise ValueError("Specify actor_rollout_ref.distillation.teacher_path.")
    if method in {"kd", "sft"} and rollout.n != 1:
        raise ValueError("Fixed-response KD/SFT requires rollout.n=1.")
