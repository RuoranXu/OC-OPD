"""Native veRL GRPO advantage and clipped policy-loss implementations."""


def group_advantages(*args, **kwargs):
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    return compute_grpo_outcome_advantage(*args, **kwargs)


def clipped_policy_loss(*args, **kwargs):
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    return compute_policy_loss_vanilla(*args, **kwargs)
