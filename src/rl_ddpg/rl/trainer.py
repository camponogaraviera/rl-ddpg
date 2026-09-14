"""Custom DDPG Training Loop."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from pathlib import Path

from rl_ddpg.rl.agent import DDPGAgent
from rl_ddpg.utils import (
    create_gif,
    plot_learning_curve,
    set_global_seed,
    train_params,
)

# Defaults will read from configs/config.cfg, so that calling .train() without the
# training hyperparameters matches the `rl_ddpg --train` command line:
_TRAIN_DEFAULTS = train_params()

# Set directories for saving plots after training:
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRAMES_DIR = PROJECT_ROOT / "assets" / "render"
GIF_DIR = PROJECT_ROOT / "assets"
PLOT_DIR = PROJECT_ROOT / "assets" / "plots"


def _validate_inputs(num_episodes: int, max_steps: int, buffer_size: int) -> None:
    """
    Validate training input parameters.

    Args:
        num_episodes: Number of episodes.
        max_steps: Max number of transitions per episode.
        buffer_size: Maximum number of transitions the replay buffer can store.

    Raises:
        ValueError: If num_episodes or max_steps is not greater than zero.
    """

    if num_episodes <= 0:
        raise ValueError("num_episodes must be greater than zero.")
    if max_steps > buffer_size:
        raise ValueError("max_steps must be less than or equal to buffer_size.")


def _run_episode(env, agent, inference: bool, seed: int | None = None):
    """
    Run a single episode.

    Args:
        env: Gym-compatible environment.
        agent: DDPG agent instance.
        inference: Inference mode (disables learning updates).
        seed: Optional random seed for reproducibility.

    Returns:
        Total episode reward and info dict.
    """
    obs, info = env.reset(seed=seed)
    terminated, truncated = False, False
    episode_reward = 0.0

    while not (terminated or truncated):
        action = agent.get_action(obs, inference=inference)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        if not inference:
            agent.store_transition(obs, action, reward, next_obs, done)
            # Learn after replay buffer warm up (at least batch_size transitions):
            agent.learn()

        obs = next_obs
        episode_reward += float(reward)

    return episode_reward, info


def _log_episode_result(episode_reward: float, info: dict) -> None:
    """
    Log episode results to the console.

    Args:
        episode_reward: Total reward for the episode.
        info: Optional dictionary containing environment-specific info.

    Returns:
        None
    """
    if info:
        print(
            f"Reward: {episode_reward:.2f}, "
            f"Ej: {info.get('ej'):.2f}, "
            f"Ec: {info.get('ec'):.2f}, "
            f"Ng: {info.get('ng'):.2f}, "
            f"Ej/Ec: {info.get('ej_ec'):.2f}, "
            f"Anharmonicity: {info.get('anharmonicity'):.2f}, "
            f"Dispersion: {info.get('dispersion'):.2f}, "
            f"T2: {info.get('t2'):.2f}, "
        )
    else:
        print(f"Reward: {episode_reward}")


def train(
    env: gym.Env,
    agent: DDPGAgent,
    num_episodes: int = _TRAIN_DEFAULTS["num_episodes"],
    max_steps: int = _TRAIN_DEFAULTS["max_steps"],
    render_flag: bool = False,
    inference_flag: bool = False,
    seed: int | None = _TRAIN_DEFAULTS["seed"],
) -> list[float]:
    """
    Train a DDPG agent in a custom environment.

    This follows the standard Gym-style pattern, where the training loop
    lives outside the agent class.

    Unset arguments will default to the values in configs/config.cfg, i.e., to the
    same values used by the `rl_ddpg --train` command line.

    Args:
        env: Gym-compatible environment.
        agent: DDPG agent instance.
        num_episodes: Number of episodes.
        max_steps: Max number of transitions per episode.
        config_path: Optional path to a config file containing TRAIN defaults.
        save_best_model: Save weights when the moving-average score improves.
        render_flag: Call env.render() at episode end.
        inference_flag: Evaluation mode (disables learning updates).
        seed: Global seed for Python, NumPy, TensorFlow, and environment resets.
            Pass None to leave the random number generators untouched.
    """

    _validate_inputs(num_episodes, max_steps, agent.buffer_size)

    # Seed the process. Note that the agent is already built at this point,
    # so seed before instantiating it to also make
    # the network weight initialization reproducible:
    if seed is not None:
        set_global_seed(seed)

    if hasattr(env, "max_steps"):
        env.max_steps = max_steps

    best_score = env.reward_range[0]  # float("-inf")
    score_history: list[float] = []

    mode = "inference" if inference_flag else "training"
    print(
        f"\nStarting {mode} for {num_episodes} episodes and {max_steps} steps each..."
    )

    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1:03d}.")

        episode_seed = None if seed is None else seed + ep
        episode_reward, info = _run_episode(
            env, agent, inference_flag, seed=episode_seed
        )

        # At the end of the episode:

        # Update score history and compute moving average:
        score_history.append(episode_reward)
        avg_score = float(np.mean(score_history[-100:]))

        # Save model if performance improves:
        if not inference_flag and avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        # Log results to console:
        _log_episode_result(episode_reward, info)

        # Render environment if enabled:
        if render_flag:
            env.render()

    # After training, plot learning curve and create GIFs if rendering was enabled:
    if render_flag:
        plot_learning_curve(score_history, PLOT_DIR)
        create_gif(
            frames_dir=FRAMES_DIR / "anharmonicity",
            output_dir=GIF_DIR,
            gif_name="anharmonicity.gif",
        )
        """
        create_gif(
            frames_dir=FRAMES_DIR / "coherence",
            output_dir=GIF_DIR,
            gif_name="coherence.gif",
        )
        """

    return score_history
