import configparser
from pathlib import Path

from rl_ddpg.envs import TransmonQubitEnv
from rl_ddpg.rl import DDPGAgent, train
from rl_ddpg.utils import (
    CONFIG_PATH,
    agent_params,
    env_params,
    load_config,
    net_params,
    set_global_seed,
    train_params,
)

__all__ = ["load_config", "setup_env_and_agent", "set_global_seed", "main"]


def setup_env_and_agent(
    config: configparser.ConfigParser, max_steps: int
) -> tuple[TransmonQubitEnv, DDPGAgent]:
    """Initialize the environment and agent.

    Args:
        config: Configuration object.
        max_steps: Maximum number of steps per episode.

    Returns:
        env: Environment instance.
        agent: Agent instance.
    """

    # Instantiate the environment:
    env = TransmonQubitEnv(**env_params(config), max_steps=max_steps)

    # Instantiate the agent:
    agent = DDPGAgent(
        env.observation_space,
        env.action_space,
        **net_params(config),
        **agent_params(config),
    )

    return env, agent


def main(
    train_flag: bool = False,
    render_flag: bool = False,
    inference_flag: bool = False,
    config_path: str | Path = CONFIG_PATH,
) -> None:
    """Main function to train or evaluate the agent.

    Args:
        train_flag: Whether to run training.
        render_flag: Whether to render the environment during the rollouts.
        inference_flag: Whether to run in inference mode (no learning updates).
        config_path: Path to the configuration file.

    Raises:
        ValueError: If neither train_flag nor inference_flag is set.

    Returns:
        None.
    """

    # Both modes share the same rollout loop, so at least one is required:
    if not (train_flag or inference_flag):
        raise ValueError(
            "Nothing to run: set train_flag=True to train, "
            "or inference_flag=True to evaluate the saved models."
        )

    # Load the configuration file:
    config = load_config(config_path)

    # Training hyperparameters:
    train_config = train_params(config)
    num_episodes = train_config["num_episodes"]
    max_steps = train_config["max_steps"]
    seed = train_config["seed"]

    set_global_seed(seed)

    # Instantiate the environment and agent:
    env, agent = setup_env_and_agent(config, max_steps)

    if inference_flag:
        agent.load_full_models()

    # Rollout loop, shared by training and inference:
    train(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        render_flag=render_flag,
        inference_flag=inference_flag,
        seed=seed,
    )


if __name__ == "__main__":
    main()
