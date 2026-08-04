from excavator_control.reference_progress import advance_reference_progress


def test_reference_progress_limits_localization_jump():
    assert advance_reference_progress(0.29, 0.68, 0.01) == 0.30


def test_reference_progress_never_regresses():
    assert advance_reference_progress(0.68, 0.29, 0.01) == 0.68


def test_reference_progress_initializes_and_reaches_target():
    progress = advance_reference_progress(None, 0.29, 0.01)
    assert progress == 0.29
    for _ in range(71):
        progress = advance_reference_progress(progress, 0.99, 0.01)
    assert progress == 0.99
