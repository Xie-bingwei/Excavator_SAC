import numpy as np

from excavator_control.terminal_state import (
    TerminalConfig,
    TerminalObservation,
    TerminalPhase,
    TerminalStateMachine,
)


def _observation(
        s_star=0.97, path_error=0.1, endpoint_error=0.1,
        q4=-0.725, q4_target=-0.725):
    q = np.array([0.0, 0.0, -1.2, q4])
    return TerminalObservation(
        s_star=s_star,
        path_error=path_error,
        endpoint_error=endpoint_error,
        q=q,
        q4_terminal_target=q4_target,
    )


def test_observed_incomplete_curl_enters_hold_not_done():
    machine = TerminalStateMachine(TerminalConfig())
    decision = machine.update(
        10.0,
        _observation(s_star=61.0 / 63.0, endpoint_error=0.1, q4=-0.141),
    )

    assert decision.phase == TerminalPhase.TERMINAL_HOLD
    assert decision.reason == 'terminal_entry'
    assert not decision.q4_reached


def test_completion_requires_endpoint_and_q4():
    machine = TerminalStateMachine(TerminalConfig())
    machine.update(0.0, _observation(endpoint_error=0.5, q4=-0.141))

    endpoint_only = machine.update(1.0, _observation(endpoint_error=0.1, q4=-0.141))
    assert endpoint_only.phase == TerminalPhase.TERMINAL_HOLD
    assert not endpoint_only.q4_reached

    q4_only = machine.update(2.0, _observation(endpoint_error=0.5, q4=-0.725))
    assert q4_only.phase == TerminalPhase.TERMINAL_HOLD
    assert not q4_only.endpoint_reached

    complete = machine.update(3.0, _observation(endpoint_error=0.1, q4=-0.725))
    assert complete.phase == TerminalPhase.DONE
    assert complete.reason == 'terminal_complete'


def test_timeout_is_time_based_and_latched():
    config = TerminalConfig(hold_timeout_sec=8.0)
    machine = TerminalStateMachine(config)
    machine.update(100.0, _observation(endpoint_error=0.5, q4=-0.141))

    before_timeout = machine.update(107.999, _observation(endpoint_error=0.5, q4=-0.141))
    assert before_timeout.phase == TerminalPhase.TERMINAL_HOLD

    timeout = machine.update(108.0, _observation(endpoint_error=0.1, q4=-0.725))
    assert timeout.phase == TerminalPhase.TERMINAL_HOLD_TIMEOUT
    assert timeout.reason == 'terminal_hold_timeout'
    assert np.isclose(timeout.frozen_q[3], -0.725)

    later = machine.update(120.0, _observation(endpoint_error=0.0, q4=-0.725))
    assert later.phase == TerminalPhase.TERMINAL_HOLD_TIMEOUT
    assert later.reason is None
    assert np.isclose(later.frozen_q[3], -0.725)
