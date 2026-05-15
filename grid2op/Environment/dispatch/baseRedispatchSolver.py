from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional

from .dispatchTypes import RedispatchConstraints, RedispatchState


class BaseRedispatchSolver(ABC):
    def __init__(self) -> None:
        self.env = None

    def bind(self, env):
        self.env = env
        return self

    def copy_for_env(self, env):
        new_obj = copy.copy(self)
        new_obj.bind(env)
        return new_obj

    @abstractmethod
    def solve(
        self,
        constraints: RedispatchConstraints,
        state: RedispatchState,
    ) -> Optional[Exception]:
        """Compute state.actual_dispatch in-place. Return None on success."""

    @abstractmethod
    def reset(self, state: RedispatchState) -> None:
        """Reset redispatch-related state."""
