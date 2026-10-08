"""Hydra entry point; --cfg job resolves recipes without allocating GPUs."""

import hydra


@hydra.main(version_base=None, config_path="config", config_name="oc_opd")
def main(config):
    import ray

    from verl.trainer.main_ppo import run_ppo

    from .task_runner import TrainingTaskRunner
    from .validation import validate_training_config

    validate_training_config(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TrainingTaskRunner))


if __name__ == "__main__":
    main()
