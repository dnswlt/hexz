from pyhexz.flagz_tail import (
    ALL_CELLS_MASK,
    CELL_INDEX,
    NUM_CELLS,
    ExactTailSolver,
    PlayerTailState,
    optimistic_reachable_cells,
    optimistic_reachable_values,
    play_normal,
)


def state_with_open_cells(
    open_cells, *, next_values=None, grass=None
) -> PlayerTailState:
    open_mask = sum(1 << CELL_INDEX[cell] for cell in open_cells)
    next_data = bytearray(NUM_CELLS)
    for cell, value in (next_values or {}).items():
        next_data[CELL_INDEX[cell]] = value
    grass_data = bytearray(NUM_CELLS)
    for cell, value in (grass or {}).items():
        grass_data[CELL_INDEX[cell]] = value
    return PlayerTailState(
        ALL_CELLS_MASK ^ open_mask,
        bytes(next_data),
        bytes(grass_data),
    )


def test_value_five_does_not_propagate_to_normal_cells():
    state = state_with_open_cells(
        [(0, 0), (0, 1), (0, 2)],
        next_values={(0, 0): 5},
    )

    reachable = optimistic_reachable_cells(state)

    assert reachable == {CELL_INDEX[(0, 0)]}


def test_grass_can_reset_propagation_after_a_five():
    state = state_with_open_cells(
        [(0, 0), (0, 2), (0, 3)],
        next_values={(0, 0): 5},
        grass={(0, 1): 1},
    )

    values = optimistic_reachable_values(state)

    assert values[CELL_INDEX[(0, 0)]] == {5}
    assert values[CELL_INDEX[(0, 1)]] == {1}
    assert values[CELL_INDEX[(0, 2)]] == {2}
    assert values[CELL_INDEX[(0, 3)]] == {3}


def test_play_normal_captures_grass_and_propagates_from_it():
    state = state_with_open_cells(
        [(0, 0), (0, 2), (0, 3)],
        next_values={(0, 0): 5},
        grass={(0, 1): 1},
    )

    result, gained = play_normal(state, CELL_INDEX[(0, 0)])

    assert gained == 6
    assert result.grass[CELL_INDEX[(0, 1)]] == 0
    assert result.next_values[CELL_INDEX[(0, 2)]] == 2


def test_exact_solver_scores_a_forced_chain():
    state = state_with_open_cells(
        [(0, 0), (0, 1), (0, 2)],
        next_values={(0, 0): 1},
    )

    solver = ExactTailSolver(node_limit=1_000, time_limit_seconds=1)

    assert solver.solve(state) == 6


def test_exact_solver_adds_independent_components():
    state = state_with_open_cells(
        [(0, 0), (0, 1), (10, 0), (10, 1)],
        next_values={(0, 0): 1, (10, 0): 3},
    )

    solver = ExactTailSolver(node_limit=1_000, time_limit_seconds=1)

    assert solver.solve(state) == 10
