package hexz

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"
)

type mcNode struct {
	r, c         int
	cellType     CellType
	children     []*mcNode
	wins         float64
	count        float64
	done         bool
	turn         int
	doneChildren int
}

func (n *mcNode) Q() float64 {
	if n.count == 0 {
		return 0
	}
	return n.wins / n.count
}

func (n *mcNode) U(parentCount float64, uctFactor float64) float64 {
	if n.count == 0.0 {
		// Never played => infinitely interesting
		return math.Inf(1)
	}
	return n.wins/n.count + uctFactor*math.Sqrt(math.Log(parentCount)/n.count)
}

func (n *mcNode) size() int {
	s := 0
	for _, c := range n.children {
		s += c.size()
	}
	return 1 + s
}

type MCTS struct {
	gameEngine       SinglePlayerGameEngine
	rnd              *rand.Rand
	MaxFlagPositions int // maximum number of (random) positions to consider for placing a flag in a single move.
	UctFactor        float64
	FlagsFirst       bool // If true, flags will be played whenever possible.
}

func (mcts *MCTS) playRandomGame(firstMove *mcNode) (winner int) {
	ge := mcts.gameEngine
	b := ge.Board()
	if !mcts.gameEngine.MakeMove(GameEngineMove{
		playerNum: firstMove.turn,
		move:      b.Move,
		row:       firstMove.r,
		col:       firstMove.c,
		cellType:  firstMove.cellType,
	}) {
		panic("Invalid move")
	}
	for !ge.IsDone() {
		m, err := ge.RandomMove()
		if err != nil {
			log.Fatalf("Could not suggest a move: %s", err.Error())
		}
		if !ge.MakeMove(m) {
			log.Fatalf("Could not make a move")
			return
		}
	}
	return ge.Winner()
}

func (mcts *MCTS) getNextByUtc(node *mcNode) *mcNode {
	var next *mcNode
	maxUct := -1.0
	for _, l := range node.children {
		if l.done {
			continue
		}
		uct := l.U(node.count, mcts.UctFactor)
		if uct > maxUct {
			next = l
			maxUct = uct
		}
	}
	return next
}

func (mcts *MCTS) nextMoves(node *mcNode, b *Board) []*mcNode {
	cs := make([]*mcNode, 0, 16)
	hasFlag := b.Resources[b.Turn-1].NumPieces[cellFlag] > 0
	maxFlags := mcts.MaxFlagPositions
	var flagMoves []*mcNode
	if hasFlag {
		flagMoves = make([]*mcNode, maxFlags)
	}
	nFlags := 0
	for r := 0; r < len(b.Fields); r++ {
		for c := 0; c < len(b.Fields[r]); c++ {
			f := &b.Fields[r][c]
			if f.occupied() {
				continue
			}
			if f.isAvail(b.Turn) {
				cs = append(cs, &mcNode{
					r: r, c: c, turn: b.Turn,
				})
			}
			if hasFlag && (mcts.rnd.Float64() < float64(maxFlags)/(float64(nFlags)+1)) {
				// reservoir sampling to pick maxFlags with equal probability among all possibilities.
				k := nFlags
				if k >= maxFlags {
					k = mcts.rnd.Intn(maxFlags)
				}
				flagMoves[k] = &mcNode{
					r: r, c: c, turn: b.Turn, cellType: cellFlag,
				}
				nFlags++
			}
		}
	}
	if nFlags > maxFlags {
		nFlags = maxFlags
	}
	if nFlags > 0 && mcts.FlagsFirst {
		// Forced play of flags
		return flagMoves[:nFlags]
	}
	if nFlags == 0 {
		return cs
	}
	return append(cs, flagMoves[:nFlags]...)
}

func (mcts *MCTS) backpropagate(path []*mcNode, winner int) {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i].turn == winner {
			path[i].wins += 1
		} else if winner == 0 {
			path[i].wins += 0.5
		}
		path[i].count += 1
	}
}

func (mcts *MCTS) run(path []*mcNode) (depth int, moved bool) {
	node := path[len(path)-1]
	ge := mcts.gameEngine
	b := mcts.gameEngine.Board()
	if node.children == nil {
		// Terminal node in our exploration graph.
		// It's not a terminal node in the whole game, though.
		cs := mcts.nextMoves(node, b)
		if len(cs) == 0 {
			// All children are duplicates
			node.done = true
			return 0, false
		}
		node.children = cs
		// Play a random child (rollout)
		c := cs[mcts.rnd.Intn(len(cs))]
		winner := mcts.playRandomGame(c)
		path = append(path, c)
		mcts.backpropagate(path, winner)
		return len(path), true
	}
	// Node has children already, descend to the one with the highest UTC.
	c := mcts.getNextByUtc(node)
	if c == nil {
		// All children are done, no point in replaying them.
		node.done = true
		return 0, false
	}
	move := GameEngineMove{
		playerNum: c.turn, move: b.Move, row: c.r, col: c.c, cellType: c.cellType,
	}
	if !ge.MakeMove(move) {
		panic("I cannot move")
	}
	path = append(path, c)
	if ge.IsDone() {
		// This was the last move. Propagate the result up.
		c.done = true
		winner := ge.Winner()
		mcts.backpropagate(path, winner)
		return len(path), true
	}
	// Descend to next level
	depth, moved = mcts.run(path)
	if c.done {
		node.doneChildren++
		if node.doneChildren == len(node.children) {
			node.done = true
		}
	}
	return
}

type MCTSMoveStats struct {
	row        int
	col        int
	cellType   CellType
	U          float64
	Q          float64
	iterations int
}

type MCTSStats struct {
	Iterations int
	MaxDepth   int
	TreeSize   int
	Elapsed    time.Duration
	Moves      []MCTSMoveStats
	NotMoved   int // Number of times we ended up in a leaf node that has no valid successors.
}

func (s *MCTSStats) MinQ() float64 {
	r := math.Inf(1)
	for _, c := range s.Moves {
		if c.Q < r {
			r = c.Q
		}
	}
	return r
}

func (s *MCTSStats) MaxQ() float64 {
	r := 0.0
	for _, c := range s.Moves {
		if c.Q > r {
			r = c.Q
		}
	}
	return r
}

func (s *MCTSStats) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "N: %d\nmaxDepth:%d\nsize:%d\nelapsed:%.3f\nN/sec:%.1f\nfailureRate:%.3f\n",
		s.Iterations, s.MaxDepth, s.TreeSize, s.Elapsed.Seconds(), float64(s.Iterations)/s.Elapsed.Seconds(), float64(s.NotMoved)/float64(s.Iterations))
	for _, m := range s.Moves {
		cellType := ""
		if m.cellType == cellFlag {
			cellType = " F"
		}
		fmt.Fprintf(&sb, "  (%d,%d%s) U:%.3f Q:%.2f N:%d\n", m.row, m.col, cellType, m.U, m.Q, m.iterations)
	}
	return sb.String()
}

func NewMCTS(g SinglePlayerGameEngine) *MCTS {
	return &MCTS{
		gameEngine:       g,
		rnd:              rand.New(rand.NewSource(time.Now().UnixNano())),
		MaxFlagPositions: 5,
		UctFactor:        1.0,
		FlagsFirst:       false,
	}
}

func (mcts *MCTS) SuggestMove(maxDuration time.Duration) (GameEngineMove, *MCTSStats) {
	origBoard := mcts.gameEngine.Board()
	root := &mcNode{turn: origBoard.Turn}
	started := time.Now()
	n := 0
	maxDepth := 0
	notMoved := 0
	for {
		// Check every N rounds if we're done.
		if n&63 == 0 && time.Since(started) >= maxDuration {
			break
		}
		mcts.gameEngine.SetBoard(origBoard.copy())
		path := make([]*mcNode, 1, 100)
		path[0] = root
		depth, moved := mcts.run(path)
		if depth > maxDepth {
			maxDepth = depth
		}
		if !moved {
			notMoved++
		}
		if root.done {
			log.Print("board completely explored")
			break
		}
		n++
	}
	elapsed := time.Since(started)
	mcts.gameEngine.SetBoard(origBoard)

	// Return some stats
	stats := &MCTSStats{
		Iterations: int(root.count),
		MaxDepth:   maxDepth,
		Elapsed:    elapsed,
		TreeSize:   root.size(),
		Moves:      make([]MCTSMoveStats, len(root.children)),
		NotMoved:   notMoved,
	}
	var best *mcNode
	for i, c := range root.children {
		if best == nil || c.Q() > best.Q() {
			best = c
		}
		stats.Moves[i] = MCTSMoveStats{
			row:        c.r,
			col:        c.c,
			cellType:   c.cellType,
			iterations: int(c.count),
			U:          c.U(root.count, mcts.UctFactor),
			Q:          c.Q(),
		}
	}
	move := GameEngineMove{
		playerNum: best.turn, move: origBoard.Move, row: best.r, col: best.c, cellType: best.cellType,
	}
	return move, stats
}
