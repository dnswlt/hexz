"""Conservative reachability and exact separated-tail solving for Flagz.

The functions in this module operate on one player's view of a board after
both players have placed all their flags.  They deliberately do not decide
whether two players are separated; callers do that by intersecting the two
players' optimistic reachable-cell sets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


ROWS = 11
VALID_CELLS = tuple(
    (r, c) for r in range(ROWS) for c in range(10 - r % 2)
)
CELL_INDEX = {cell: index for index, cell in enumerate(VALID_CELLS)}
NUM_CELLS = len(VALID_CELLS)
ALL_CELLS_MASK = (1 << NUM_CELLS) - 1


def _neighbors(r: int, c: int) -> tuple[int, ...]:
    shift = r % 2
    candidates = (
        (r, c + 1),
        (r - 1, c + shift),
        (r - 1, c - 1 + shift),
        (r, c - 1),
        (r + 1, c - 1 + shift),
        (r + 1, c + shift),
    )
    return tuple(
        CELL_INDEX[cell] for cell in candidates if cell in CELL_INDEX
    )


NEIGHBORS = tuple(_neighbors(*cell) for cell in VALID_CELLS)


@dataclass(frozen=True, slots=True)
class PlayerTailState:
    """The mutable part of a board relevant to one player's future moves."""

    blocked: int
    next_values: bytes
    grass: bytes

    def __post_init__(self) -> None:
        if len(self.next_values) != NUM_CELLS:
            raise ValueError("next_values has the wrong size")
        if len(self.grass) != NUM_CELLS:
            raise ValueError("grass has the wrong size")

    def legal_moves(self) -> tuple[int, ...]:
        return tuple(i for i, value in enumerate(self.next_values) if value)


@dataclass(frozen=True, slots=True)
class Reachability:
    values: tuple[frozenset[int], ...]
    path_states: int
    used_relaxed_fallback: bool

    @property
    def cells(self) -> frozenset[int]:
        return frozenset(
            cell for cell, values in enumerate(self.values) if values
        )


def _relaxed_reachable_values(
    state: PlayerTailState,
) -> tuple[frozenset[int], ...]:
    values: list[set[int]] = [set() for _ in range(NUM_CELLS)]
    pending = [
        (cell, value)
        for cell, value in enumerate(state.next_values)
        if value
    ]
    while pending:
        cell, value = pending.pop()
        if value in values[cell]:
            continue
        values[cell].add(value)

        for neighbor in NEIGHBORS[cell]:
            grass_value = state.grass[neighbor]
            if grass_value and grass_value <= value:
                pending.append((neighbor, grass_value))

            if value >= 5 or state.blocked & (1 << neighbor):
                continue
            propagated = value + 1
            existing = state.next_values[neighbor]
            if existing:
                propagated = min(propagated, existing)
            pending.append((neighbor, propagated))

    return tuple(frozenset(cell_values) for cell_values in values)


def optimistic_reachability(
    state: PlayerTailState, *, path_state_limit: int = 100_000
) -> Reachability:
    """Returns conservative reachability using simple causal move paths.

    Every cell that becomes playable has a causal path from a currently legal
    move.  Cells cannot be occupied twice, so paths with repeated cells are
    unnecessary.  Tracking the used-cell mask prevents artificial cycles from
    extending a 1..5 sequence indefinitely.

    For a fixed ``(cell, value)``, a path whose used-cell mask is a subset of
    another path's mask dominates it.  If the number of non-dominated path
    states exceeds ``path_state_limit``, the function safely falls back to a
    looser finite-state propagation that may only add reachable cells.
    """

    if path_state_limit <= 0:
        raise ValueError("path_state_limit must be positive")
    values: list[set[int]] = [set() for _ in range(NUM_CELLS)]
    used_by_cell_value: dict[tuple[int, int], list[int]] = {}
    pending: list[tuple[int, int, int]] = []
    path_states = 0

    def add(cell: int, value: int, used: int) -> bool:
        nonlocal path_states
        key = (cell, value)
        masks = used_by_cell_value.setdefault(key, [])
        if any(existing & used == existing for existing in masks):
            return True
        masks[:] = [
            existing for existing in masks if used & existing != used
        ]
        masks.append(used)
        pending.append((cell, value, used))
        path_states += 1
        return path_states <= path_state_limit

    for cell, value in enumerate(state.next_values):
        if value and not add(cell, value, 1 << cell):
            return Reachability(
                _relaxed_reachable_values(state), path_states, True
            )

    while pending:
        cell, value, used = pending.pop()
        values[cell].add(value)
        for neighbor in NEIGHBORS[cell]:
            neighbor_bit = 1 << neighbor
            if used & neighbor_bit:
                continue

            grass_value = state.grass[neighbor]
            if grass_value and grass_value <= value:
                if not add(
                    neighbor, grass_value, used | neighbor_bit
                ):
                    return Reachability(
                        _relaxed_reachable_values(state), path_states, True
                    )

            if value >= 5 or state.blocked & neighbor_bit:
                continue
            propagated = value + 1
            existing = state.next_values[neighbor]
            if existing:
                propagated = min(propagated, existing)
            if not add(neighbor, propagated, used | neighbor_bit):
                return Reachability(
                    _relaxed_reachable_values(state), path_states, True
                )

    return Reachability(
        tuple(frozenset(cell_values) for cell_values in values),
        path_states,
        False,
    )


def optimistic_reachable_values(
    state: PlayerTailState,
) -> tuple[frozenset[int], ...]:
    return optimistic_reachability(state).values


def optimistic_reachable_cells(state: PlayerTailState) -> frozenset[int]:
    return optimistic_reachability(state).cells


def _play_normal_mutable(
    blocked: int,
    next_values: bytearray,
    grass: bytearray,
    cell: int,
) -> tuple[int, int]:
    """Applies one normal move and all automatic grass captures."""

    value = next_values[cell]
    if not value or blocked & (1 << cell):
        raise ValueError(f"cell {cell} is not a legal normal move")

    blocked |= 1 << cell
    next_values[cell] = 0
    if value < 5:
        propagated = value + 1
        for neighbor in NEIGHBORS[cell]:
            if blocked & (1 << neighbor):
                continue
            existing = next_values[neighbor]
            if not existing or existing > propagated:
                next_values[neighbor] = propagated
    else:
        for neighbor in NEIGHBORS[cell]:
            blocked |= 1 << neighbor
            next_values[neighbor] = 0

    gained = value
    for neighbor in NEIGHBORS[cell]:
        grass_value = grass[neighbor]
        if not grass_value or grass_value > value:
            continue
        grass[neighbor] = 0
        blocked &= ~(1 << neighbor)
        next_values[neighbor] = grass_value
        blocked, captured = _play_normal_mutable(
            blocked, next_values, grass, neighbor
        )
        gained += captured
    return blocked, gained


def play_normal(
    state: PlayerTailState, cell: int
) -> tuple[PlayerTailState, int]:
    """Returns the state and score gain after a legal normal move."""

    next_values = bytearray(state.next_values)
    grass = bytearray(state.grass)
    blocked, gained = _play_normal_mutable(
        state.blocked, next_values, grass, cell
    )
    return (
        PlayerTailState(blocked, bytes(next_values), bytes(grass)),
        gained,
    )


def reachable_components(state: PlayerTailState) -> tuple[frozenset[int], ...]:
    """Partitions optimistic reachable cells into independent components."""

    remaining = set(optimistic_reachable_cells(state))
    components = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        pending = [seed]
        while pending:
            cell = pending.pop()
            for neighbor in NEIGHBORS[cell]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(frozenset(component))
    return tuple(components)


def restrict_to_component(
    state: PlayerTailState, component: frozenset[int]
) -> PlayerTailState:
    allowed = sum(1 << cell for cell in component)
    next_values = bytes(
        value if cell in component else 0
        for cell, value in enumerate(state.next_values)
    )
    grass = bytes(
        value if cell in component else 0
        for cell, value in enumerate(state.grass)
    )
    return PlayerTailState(
        state.blocked | (ALL_CELLS_MASK ^ allowed),
        next_values,
        grass,
    )


class TailSolveLimit(RuntimeError):
    """Raised when an exact tail solve exceeds its configured budget."""


@dataclass(slots=True)
class TailSolveStats:
    states: int = 0
    cache_hits: int = 0
    branches: int = 0
    upper_bound_prunes: int = 0
    component_sizes: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class ExactTailSolver:
    """Memoized single-player maximization with conservative pruning."""

    def __init__(
        self,
        *,
        node_limit: int = 250_000,
        time_limit_seconds: float = 1.0,
    ) -> None:
        if node_limit <= 0:
            raise ValueError("node_limit must be positive")
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        self.node_limit = node_limit
        self.time_limit_seconds = time_limit_seconds
        self.stats = TailSolveStats()
        self._memo: dict[PlayerTailState, int] = {}
        self._upper_bounds: dict[PlayerTailState, int] = {}
        self._deadline = 0.0

    def solve(self, state: PlayerTailState) -> int:
        started = time.monotonic()
        self._deadline = started + self.time_limit_seconds
        result = 0
        try:
            components = reachable_components(state)
            self.stats.component_sizes.extend(
                sorted(
                    (len(component) for component in components),
                    reverse=True,
                )
            )
            for component in components:
                result += self._solve_state(
                    restrict_to_component(state, component)
                )
            return result
        finally:
            self.stats.elapsed_seconds = time.monotonic() - started

    def _check_budget(self) -> None:
        if self.stats.states >= self.node_limit:
            raise TailSolveLimit(
                f"tail solver exceeded {self.node_limit} states"
            )
        if self.stats.states % 256 == 0 and time.monotonic() > self._deadline:
            raise TailSolveLimit(
                "tail solver exceeded "
                f"{self.time_limit_seconds:.3f} seconds"
            )

    def _upper_bound(self, state: PlayerTailState) -> int:
        cached = self._upper_bounds.get(state)
        if cached is not None:
            return cached
        reachable = _relaxed_reachable_values(state)
        result = sum(max(values) for values in reachable if values)
        self._upper_bounds[state] = result
        return result

    def _solve_state(self, state: PlayerTailState) -> int:
        cached = self._memo.get(state)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        self._check_budget()
        self.stats.states += 1
        moves = state.legal_moves()
        if not moves:
            self._memo[state] = 0
            return 0

        children = []
        for cell in moves:
            child, gained = play_normal(state, cell)
            children.append((gained, cell, child))
        children.sort(reverse=True, key=lambda item: (item[0], item[1]))

        best = 0
        for gained, _, child in children:
            self.stats.branches += 1
            if gained + self._upper_bound(child) <= best:
                self.stats.upper_bound_prunes += 1
                continue
            best = max(best, gained + self._solve_state(child))
            if best >= self._upper_bound(state):
                break
        self._memo[state] = best
        return best
