"""RLlib new-stack PPO RLModule with action masking for Sternhalma."""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

torch, _ = try_import_torch()


class SternhalmaActionMaskingTorchRLModule(DefaultPPOTorchRLModule):
    """PPO RLModule that masks invalid actions using `obs['action_mask']`."""

    @override(DefaultPPOTorchRLModule)
    def __init__(
        self,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[dict[str, Any]] = None,
        catalog_class=None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                "SternhalmaActionMaskingTorchRLModule requires Dict observations "
                "with keys: 'observations' and 'action_mask'."
            )
        if "observations" not in observation_space.spaces or "action_mask" not in observation_space.spaces:
            raise ValueError(
                "Observation space must contain both 'observations' and 'action_mask'."
            )

        super().__init__(
            observation_space=observation_space["observations"],
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )
        self._checked_batch_once = False

    @override(DefaultPPOTorchRLModule)
    def _forward(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        action_mask, batch_wo_mask = self._extract_mask_and_obs(batch)
        outputs = super()._forward(batch_wo_mask, **kwargs)
        return self._mask_action_logits(outputs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        action_mask, batch_wo_mask = self._extract_mask_and_obs(batch)
        outputs = super()._forward_train(batch_wo_mask, **kwargs)
        return self._mask_action_logits(outputs, action_mask)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        if embeddings is not None:
            return super().compute_values(batch, embeddings)

        if isinstance(batch.get(Columns.OBS), dict):
            _, batch_wo_mask = self._extract_mask_and_obs(batch)
            return super().compute_values(batch_wo_mask, embeddings)

        return super().compute_values(batch, embeddings)

    def _extract_mask_and_obs(
        self, batch: Dict[str, TensorType]
    ) -> tuple[TensorType, Dict[str, TensorType]]:
        self._check_batch_obs(batch)

        obs = dict(batch[Columns.OBS])
        action_mask = obs["action_mask"]
        observations = obs["observations"]

        processed = batch.copy()
        processed[Columns.OBS] = observations
        return action_mask, processed

    def _mask_action_logits(
        self, outputs: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        logits = outputs[Columns.ACTION_DIST_INPUTS]
        mask = action_mask.to(dtype=logits.dtype)
        has_legal = torch.sum(mask, dim=-1, keepdim=True) > 0
        safe_mask = torch.where(has_legal, mask, torch.ones_like(mask))
        inf_mask = torch.clamp(torch.log(safe_mask), min=FLOAT_MIN)
        outputs[Columns.ACTION_DIST_INPUTS] = logits + inf_mask
        return outputs

    def _check_batch_obs(self, batch: Dict[str, TensorType]) -> None:
        if self._checked_batch_once:
            return
        obs = batch.get(Columns.OBS)
        if not isinstance(obs, dict):
            raise ValueError("Expected dict observations with action mask.")
        if "action_mask" not in obs:
            raise ValueError("Missing 'action_mask' in observation batch.")
        if "observations" not in obs:
            raise ValueError("Missing 'observations' in observation batch.")
        self._checked_batch_once = True
