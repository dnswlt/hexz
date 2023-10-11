import numpy as np

    
def c_valid_idx(r_c):
    """Returns True if (r, c) = r_c represents a valid hexz board index."""
    r, c = r_c
    return r >= 0 and r < 11 and c >= 0 and c < 10-r%2


def c_neighbors_map():
    """Returns a dict mapping all valid (r, c) indices to their neighbor indices.
    
    The neighbor indices are represented as (row, column) tuples."""
    result = {}
    for r in range(11):
        shift = r%2 # Depending on the row, neighbors below and above are shifted.
        for c in range(10-r%2):
            ns = filter(c_valid_idx, [
                (r, c+1),
                (r-1, c+shift),
                (r-1, c-1+shift),
                (r, c-1),
                (r+1, c-1+shift),
                (r+1, c+shift),
            ])
            nr, nc = zip(*ns)  # unzip
            result[(r, c)] = (np.array(nr), np.array(nc))
            
    return result


_C_NEIGHBORS = c_neighbors_map()


def c_occupy_grass(board, int player, int r, int c):
    """Occupies the neighboring grass cells of move_idx (a 3-tuple index into a move) for player.

    Expects that the move has already been played.
    """
    cdef float[:, :, :] b = board.b
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t x = 0
    cdef Py_ssize_t y = 0
    
    nx, ny = _C_NEIGHBORS[(r, c)]
    for i in range(len(nx)):
        x = nx[i]
        y = ny[i]
        grass_val = b[8, x, y]
        if grass_val > 0 and grass_val <= b[1 + player*4, r, c]:
            # Occupy: first remove grass
            b[8, x, y] = 0                
            # the rest is exactly like playing a move.
            c_make_move(board, player, (1, r, c, grass_val))


def c_make_move(board, int player, move):
    """Makes the given move.

    Args:
      board: the board as an (N, 11, 10) ndarray.
      player: 0 or 1
      move: a 4-tuple of (typ, r, c, val), where typ = 0 (flag) or 1 (normal)
    Does not check that it is a valid move. Should be called only
    with moves returned from `next_moves`.
    """
    cdef float[:, :, :] b = board.b
    cdef Py_ssize_t typ = move[0]
    cdef Py_ssize_t r = move[1]
    cdef Py_ssize_t c = move[2]
    cdef double val = move[3]
    cdef double next_val = 0
    cdef Py_ssize_t i = 0
    
    b[typ + player*4, r, c] = val
    played_flag = typ == 0
    # Block played cell for both players.
    b[2, r, c] = 1
    b[6, r, c] = 1
    # Set next value to 0 for occupied cell.
    b[3, r, c] = 0
    b[7, r, c] = 0
    # Block neighboring cells if a 5 was played.
    nx, ny = _C_NEIGHBORS[(r, c)]
    # Update next value of neighboring cells. If we played a flag, the next value is 1.
    if played_flag:
        next_val = 1
        board.nflags[player] -= 1
    else:
        next_val = val + 1
    if next_val <= 5:
        for nr, nc in zip(nx, ny):
            if b[2 + player*4, nr, nc] == 0:
                if next_val > 5:
                    b[3 + player*4, nr, nc] = 0
                if b[3 + player*4, nr, nc] == 0:
                    b[3 + player*4, nr, nc] = next_val
                elif b[3 + player*4, nr, nc] > next_val:
                    b[3 + player*4, nr, nc] = next_val
    else:
        for i in range(len(nx)):
            # Played a 5: block neighboring cells and clear next value.
            b[2 + player*4, nx[i], ny[i]] = 1
            b[3 + player*4, nx[i], ny[i]] = 0  # Clear next value.
    if not played_flag:
        c_occupy_grass(board, player, r, c)
