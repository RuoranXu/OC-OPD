"""Actor/rollout worker with a separate, frozen, sharded online teacher."""

from omegaconf import open_dict

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .actor import DistillationActor


class DistillationWorker(ActorRolloutRefWorker):
    def __init__(self, config, role, **kwargs):
        # KD/SFT use fixed responses and need no vLLM model or rollout KV cache.
        offline = config.distillation.name in {"kd", "sft"}
        super().__init__(config, "actor" if offline else role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_lora:
            self._validate_student_adapter()
        teacher = None
        if self.config.distillation.name != "sft":
            # Upstream model construction sets these worker attributes. Keep
            # actor generation and checkpoint tokenizers attached to the student.
            saved = {
                k: getattr(self, k)
                for k in (
                    "tokenizer",
                    "processor",
                    "generation_config",
                    "use_orig_params",
                    "_is_lora",
                    "_lora_rank",
                )
            }
            adapter_settings = {
                key: self.config.model.get(key) for key in ("lora_rank", "lora_adapter_path")
            }
            try:
                # veRL's builder applies the worker's LoRA settings even when
                # role="ref". The teacher must never inherit a student adapter.
                self._is_lora = False
                self._lora_rank = 0
                with open_dict(self.config.model):
                    self.config.model.lora_rank = 0
                    self.config.model.lora_adapter_path = None
                teacher, _, _, teacher_config = self._build_model_optimizer(
                    model_path=copy_to_local(self.config.distillation.teacher_path),
                    fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                    optim_config=None,
                    override_model_config={
                        "attn_implementation": self.config.distillation.teacher_attention
                    },
                    trust_remote_code=self.config.model.trust_remote_code,
                    role="ref",
                )
            finally:
                with open_dict(self.config.model):
                    for key, value in adapter_settings.items():
                        setattr(self.config.model, key, value)
                for key, value in saved.items():
                    setattr(self, key, value)
            if teacher_config.vocab_size != self.actor_model_config.vocab_size:
                raise ValueError("Teacher/student output vocab sizes differ.")
            unwrapped_teacher = getattr(teacher, "_fsdp_wrapped_module", teacher)
            if hasattr(unwrapped_teacher, "peft_config"):
                raise ValueError("The frozen teacher must not contain student LoRA adapters.")
            teacher.requires_grad_(False)
            teacher.eval()
        self.actor = DistillationActor(
            self.actor.config,
            self.actor_module_fsdp,
            self.actor_optimizer,
            teacher=teacher,
            method_config=self.config.distillation,
        )

    def _validate_student_adapter(self):
        """Reject resumed adapters incompatible with the configured rollout."""
        student = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        adapter = student.peft_config.get("default")
        model = self.config.model
        if (
            adapter is None
            or adapter.r != model.lora_rank
            or adapter.lora_alpha != model.lora_alpha
            or set(adapter.target_modules) != set(model.target_modules)
            or adapter.bias != "none"
            or getattr(adapter, "modules_to_save", None)
            or getattr(adapter, "use_dora", False)
        ):
            raise ValueError(
                "Student adapter must match the configured plain LoRA rank, alpha and targets."
            )
        if not any(parameter.requires_grad for parameter in student.parameters()):
            raise ValueError("The student adapter has no trainable parameters.")
