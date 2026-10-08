"""Actor/rollout worker with a separate, frozen, sharded online teacher."""

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
        teacher = None
        if self.config.distillation.name != "sft":
            # Upstream model construction sets these worker attributes. Keep
            # actor generation and checkpoint tokenizers attached to the student.
            saved = {k: getattr(self, k) for k in ("tokenizer", "processor", "generation_config")}
            try:
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
                for key, value in saved.items():
                    setattr(self, key, value)
            if teacher_config.vocab_size != self.actor_model_config.vocab_size:
                raise ValueError("Teacher/student output vocab sizes differ.")
            teacher.requires_grad_(False)
            teacher.eval()
        self.actor = DistillationActor(
            self.actor.config,
            self.actor_module_fsdp,
            self.actor_optimizer,
            teacher=teacher,
            method_config=self.config.distillation,
        )
