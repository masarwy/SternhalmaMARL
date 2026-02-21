"""RLlib integration helpers for Sternhalma training entrypoints."""

from __future__ import annotations

import logging
from numbers import Real
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from training.models.action_mask_rl_module import SternhalmaActionMaskingTorchRLModule
from training.rllib_env import SternhalmaRLlibObsWrapper
from training.rllib_logger import MinimalRayLogger
from training.utils import ensure_dir, make_env


_RAY_LOG_FILTERS_CONFIGURED = False


class _RayWarningFilter(logging.Filter):
    _SUPPRESSED = (
        "You are running PPO on the new API stack! This is the new default behavior for this algorithm.",
        "DeprecationWarning: `RLModule(config=[RLModuleConfig object])` has been deprecated.",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(text in message for text in self._SUPPRESSED)


def _configure_ray_log_filters() -> None:
    global _RAY_LOG_FILTERS_CONFIGURED
    if _RAY_LOG_FILTERS_CONFIGURED:
        return
    warning_filter = _RayWarningFilter()
    logging.getLogger("ray.rllib.algorithms.algorithm_config").addFilter(warning_filter)
    logging.getLogger("ray.rllib.core.rl_module.rl_module").addFilter(warning_filter)
    logging.getLogger("ray._common.deprecation").addFilter(warning_filter)
    _RAY_LOG_FILTERS_CONFIGURED = True


@dataclass
class AgentSpaceInfo:
    possible_agents: list[str]
    observation_spaces: dict[str, Any]
    action_spaces: dict[str, Any]


def init_ray(ray_config: dict[str, Any] | None = None) -> bool:
    """Initialize Ray once and return whether this call started it."""
    if ray.is_initialized():
        return False

    cfg = dict(ray_config or {})
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    _configure_ray_log_filters()
    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": bool(cfg.pop("log_to_driver", False)),
        "local_mode": bool(cfg.pop("local_mode", False)),
    }
    if "num_cpus" in cfg:
        init_kwargs["num_cpus"] = int(cfg.pop("num_cpus"))
    init_kwargs.update(cfg)
    ray.init(**init_kwargs)
    return True


def shutdown_ray(started_here: bool) -> None:
    if started_here and ray.is_initialized():
        ray.shutdown()


def register_sternhalma_env(env_config: dict[str, Any], name_prefix: str) -> str:
    """
    Register a unique RLlib env ID that wraps Sternhalma's PettingZoo env.
    """
    del env_config  # env_config is passed at runtime by RLlib.
    env_name = f"{name_prefix}_{uuid.uuid4().hex[:10]}"
    register_env(env_name, lambda cfg: PettingZooEnv(SternhalmaRLlibObsWrapper(make_env(cfg))))
    return env_name


def get_agent_space_info(env_config: dict[str, Any]) -> AgentSpaceInfo:
    env = SternhalmaRLlibObsWrapper(make_env(env_config))
    try:
        possible_agents = list(env.possible_agents)
        obs_spaces = {agent_id: env.observation_space(agent_id) for agent_id in possible_agents}
        act_spaces = {agent_id: env.action_space(agent_id) for agent_id in possible_agents}
        return AgentSpaceInfo(
            possible_agents=possible_agents,
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
        )
    finally:
        env.close()


def build_ppo_config(
    env_name: str,
    env_config: dict[str, Any],
    train_cfg: dict[str, Any],
    rllib_cfg: dict[str, Any],
    seed: int,
) -> PPOConfig:
    framework = str(rllib_cfg.get("framework", "torch"))
    use_new_api_stack = bool(rllib_cfg.get("use_new_api_stack", True))
    if not use_new_api_stack:
        raise ValueError(
            "Deprecated RLlib old API stack is disabled. "
            "Set rllib_config.use_new_api_stack=true."
        )

    config = PPOConfig().framework(framework).environment(env=env_name, env_config=dict(env_config))
    config = config.api_stack(
        enable_rl_module_and_learner=use_new_api_stack,
        enable_env_runner_and_connector_v2=use_new_api_stack,
    )
    logger_config_raw = rllib_cfg.get("logger_config", {})
    logger_config = dict(logger_config_raw) if isinstance(logger_config_raw, dict) else {}
    logger_config.setdefault("type", MinimalRayLogger)

    config = config.debugging(
        log_level=str(rllib_cfg.get("log_level", "WARN")),
        seed=seed,
        log_sys_usage=bool(rllib_cfg.get("log_sys_usage", False)),
        logger_config=logger_config,
    )
    config = config.resources(
        num_gpus=float(rllib_cfg.get("num_gpus", 0.0)),
        num_cpus_for_main_process=int(rllib_cfg.get("num_cpus_for_main_process", 1)),
    )
    config = config.env_runners(
        num_env_runners=int(rllib_cfg.get("num_env_runners", 0)),
        num_envs_per_env_runner=int(rllib_cfg.get("num_envs_per_env_runner", 1)),
        rollout_fragment_length=rllib_cfg.get("rollout_fragment_length", "auto"),
        batch_mode=str(rllib_cfg.get("batch_mode", "truncate_episodes")),
    )

    # Support both the old names and the latest RLlib API names.
    train_batch_size = int(rllib_cfg.get("train_batch_size", train_cfg.get("train_batch_size", 2048)))
    minibatch_size = int(
        rllib_cfg.get(
            "minibatch_size",
            train_cfg.get("sgd_minibatch_size", train_cfg.get("minibatch_size", 256)),
        )
    )
    num_epochs = int(rllib_cfg.get("num_epochs", train_cfg.get("num_sgd_iter", train_cfg.get("num_epochs", 5))))
    legacy_model_cfg = dict(rllib_cfg.get("model", {}))
    module_model_cfg = dict(rllib_cfg.get("model_config", {}))
    module_model_cfg.setdefault(
        "fcnet_hiddens",
        legacy_model_cfg.get("fcnet_hiddens", rllib_cfg.get("fcnet_hiddens", [256, 256])),
    )
    module_model_cfg.setdefault(
        "fcnet_activation",
        legacy_model_cfg.get("fcnet_activation", rllib_cfg.get("fcnet_activation", "tanh")),
    )
    if "head_fcnet_hiddens" in legacy_model_cfg or "head_fcnet_hiddens" in rllib_cfg:
        module_model_cfg.setdefault(
            "head_fcnet_hiddens",
            legacy_model_cfg.get("head_fcnet_hiddens", rllib_cfg.get("head_fcnet_hiddens")),
        )
    if "head_fcnet_activation" in legacy_model_cfg or "head_fcnet_activation" in rllib_cfg:
        module_model_cfg.setdefault(
            "head_fcnet_activation",
            legacy_model_cfg.get("head_fcnet_activation", rllib_cfg.get("head_fcnet_activation")),
        )
    if "vf_share_layers" in legacy_model_cfg or "vf_share_layers" in rllib_cfg:
        module_model_cfg.setdefault(
            "vf_share_layers",
            bool(legacy_model_cfg.get("vf_share_layers", rllib_cfg.get("vf_share_layers"))),
        )
    config = config.rl_module(
        rl_module_spec=RLModuleSpec(module_class=SternhalmaActionMaskingTorchRLModule),
        model_config=module_model_cfg,
    )

    training_kwargs: dict[str, Any] = {
        "lr": float(rllib_cfg.get("lr", train_cfg.get("lr", 5e-5))),
        "gamma": float(rllib_cfg.get("gamma", train_cfg.get("gamma", 0.99))),
        "lambda_": float(rllib_cfg.get("lambda", train_cfg.get("lambda", 0.95))),
        "clip_param": float(rllib_cfg.get("clip_param", train_cfg.get("clip_param", 0.2))),
        "vf_loss_coeff": float(rllib_cfg.get("vf_loss_coeff", train_cfg.get("vf_loss_coeff", 1.0))),
        "entropy_coeff": float(rllib_cfg.get("entropy_coeff", train_cfg.get("entropy_coeff", 0.0))),
        "train_batch_size": train_batch_size,
        "minibatch_size": minibatch_size,
        "num_epochs": num_epochs,
    }
    if "grad_clip" in rllib_cfg or "grad_clip" in train_cfg:
        training_kwargs["grad_clip"] = float(rllib_cfg.get("grad_clip", train_cfg.get("grad_clip")))
    config = config.training(**training_kwargs)
    return config


def resolve_checkpoint_path(save_result: Any) -> str:
    """Convert Algorithm.save() return value to a path-like string."""
    checkpoint = getattr(save_result, "checkpoint", save_result)
    if checkpoint is None:
        return ""
    if hasattr(checkpoint, "path"):
        try:
            return str(checkpoint.path)
        except Exception:
            pass
    if hasattr(checkpoint, "to_directory"):
        try:
            return str(checkpoint.to_directory())
        except Exception:
            pass
    try:
        return os.fspath(checkpoint)
    except TypeError:
        rendered = str(checkpoint)
        match = re.search(r"path=([^)]+)", rendered)
        return match.group(1) if match else rendered


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    return None


def _find_first_numeric(payload: Any, candidate_keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload:
                numeric = _as_float(payload[key])
                if numeric is not None:
                    return numeric
        for value in payload.values():
            found = _find_first_numeric(value, candidate_keys)
            if found is not None:
                return found
    return None


def extract_iteration_record(result: dict[str, Any], iteration: int) -> dict[str, Any]:
    learner_results = result.get("learners", {})

    episode_return_mean = _find_first_numeric(
        result,
        ("episode_return_mean", "episode_reward_mean"),
    )
    episode_len_mean = _find_first_numeric(
        result,
        ("episode_len_mean",),
    )
    episodes_this_iter = _find_first_numeric(
        result,
        ("num_episodes", "episodes_this_iter"),
    )
    sampled_lifetime = _find_first_numeric(
        result,
        ("num_env_steps_sampled_lifetime",),
    )
    trained_lifetime = _find_first_numeric(
        result,
        ("num_env_steps_trained_lifetime",),
    )
    timesteps_total = _find_first_numeric(
        result,
        ("timesteps_total",),
    )
    episode_return_mean_val = episode_return_mean if episode_return_mean is not None else 0.0
    episode_len_mean_val = episode_len_mean if episode_len_mean is not None else 0.0
    if episode_len_mean_val > 0.0:
        episode_return_per_step = episode_return_mean_val / episode_len_mean_val
    else:
        episode_return_per_step = 0.0

    return {
        "iteration": iteration,
        "episode_return_mean": episode_return_mean_val,
        "episode_len_mean": episode_len_mean_val,
        "episode_return_per_step": episode_return_per_step,
        "episodes_this_iter": int(episodes_this_iter) if episodes_this_iter is not None else 0,
        "num_env_steps_sampled_lifetime": sampled_lifetime if sampled_lifetime is not None else 0.0,
        "num_env_steps_trained_lifetime": trained_lifetime if trained_lifetime is not None else 0.0,
        "timesteps_total": timesteps_total if timesteps_total is not None else 0.0,
        "learner_results": learner_results,
    }


def write_checkpoint_metadata(
    checkpoints_dir: Path,
    iteration: int,
    checkpoint_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dir(checkpoints_dir)
    metadata: dict[str, Any] = {
        "iteration": iteration,
        "checkpoint_path": checkpoint_path,
    }
    if extra:
        metadata.update(extra)
    return metadata


def build_mappo_multiagent(
    agent_space_info: AgentSpaceInfo,
) -> tuple[dict[str, PolicySpec], Any]:
    shared_id = "shared_policy"
    policies = {
        shared_id: PolicySpec(
            observation_space=agent_space_info.observation_spaces[agent_space_info.possible_agents[0]],
            action_space=agent_space_info.action_spaces[agent_space_info.possible_agents[0]],
        )
    }

    def mapping_fn(agent_id: str, *args: Any, **kwargs: Any) -> str:
        del agent_id, args, kwargs
        return shared_id

    return policies, mapping_fn


def build_ippo_multiagent(
    agent_space_info: AgentSpaceInfo,
) -> tuple[dict[str, PolicySpec], Any]:
    policies: dict[str, PolicySpec] = {}
    policy_by_agent: dict[str, str] = {}

    for agent_id in agent_space_info.possible_agents:
        policy_id = f"policy_{agent_id}"
        policy_by_agent[agent_id] = policy_id
        policies[policy_id] = PolicySpec(
            observation_space=agent_space_info.observation_spaces[agent_id],
            action_space=agent_space_info.action_spaces[agent_id],
        )

    def mapping_fn(agent_id: str, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return policy_by_agent[agent_id]

    return policies, mapping_fn
