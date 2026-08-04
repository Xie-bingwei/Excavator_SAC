"""Terminal curl state machine for one-shot excavator operations."""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TerminalPhase(str, Enum):
    """One-shot task phases after the path tracker approaches its end."""

    FOLLOW_PATH = 'follow_path'
    TERMINAL_HOLD = 'terminal_hold'
    DONE = 'done'
    TERMINAL_HOLD_TIMEOUT = 'terminal_hold_timeout'


@dataclass(frozen=True)
class TerminalConfig:
    """Terminal transition thresholds in SI units."""

    entry_s_threshold: float = 0.93
    entry_path_error_threshold: float = 0.35
    endpoint_error_threshold: float = 0.30
    q4_error_threshold: float = 0.08
    hold_timeout_sec: float = 8.0


@dataclass(frozen=True)
class TerminalObservation:
    """Measurements used for one terminal-state update."""

    s_star: float
    path_error: float
    endpoint_error: float
    q: np.ndarray
    q4_terminal_target: float


@dataclass(frozen=True)
class TerminalDecision:
    """Controller action requested by the terminal state machine."""

    phase: TerminalPhase
    transitioned: bool
    reason: str | None
    hold_elapsed_sec: float
    force_terminal_target: bool
    frozen_q: np.ndarray | None
    endpoint_reached: bool
    q4_reached: bool


class TerminalStateMachine:
    """Latch terminal curl completion or a safe terminal-hold timeout."""

    def __init__(self, config: TerminalConfig):
        self.config = config
        self.phase = TerminalPhase.FOLLOW_PATH
        self._hold_started_at: float | None = None
        self._frozen_q: np.ndarray | None = None

    def update(self, now_sec: float, observation: TerminalObservation) -> TerminalDecision:
        """Advance the state machine using a monotonic timestamp."""
        endpoint_reached = (
            observation.endpoint_error <= self.config.endpoint_error_threshold
        )
        q4_reached = (
            abs(observation.q[3] - observation.q4_terminal_target)
            <= self.config.q4_error_threshold
        )
        transitioned = False
        reason = None

        if self.phase == TerminalPhase.FOLLOW_PATH:
            entry_reached = (
                observation.s_star >= self.config.entry_s_threshold
                and observation.path_error <= self.config.entry_path_error_threshold
            )
            if entry_reached:
                self.phase = TerminalPhase.TERMINAL_HOLD
                self._hold_started_at = now_sec
                transitioned = True
                reason = 'terminal_entry'

        if self.phase == TerminalPhase.TERMINAL_HOLD:
            hold_elapsed = now_sec - self._hold_started_at
            if hold_elapsed >= self.config.hold_timeout_sec:
                self.phase = TerminalPhase.TERMINAL_HOLD_TIMEOUT
                self._frozen_q = observation.q.copy()
                transitioned = True
                reason = 'terminal_hold_timeout'
            elif endpoint_reached and q4_reached:
                self.phase = TerminalPhase.DONE
                transitioned = True
                reason = 'terminal_complete'

        hold_elapsed = 0.0
        if self._hold_started_at is not None:
            hold_elapsed = max(0.0, now_sec - self._hold_started_at)

        return TerminalDecision(
            phase=self.phase,
            transitioned=transitioned,
            reason=reason,
            hold_elapsed_sec=hold_elapsed,
            force_terminal_target=self.phase == TerminalPhase.TERMINAL_HOLD,
            frozen_q=None if self._frozen_q is None else self._frozen_q.copy(),
            endpoint_reached=endpoint_reached,
            q4_reached=q4_reached,
        )
