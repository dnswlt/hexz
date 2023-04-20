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
	liveChildren int // Number of child nodes that are not done yet.
}

func (n *mcNode) String() string {
	return fmt.Sprintf("(%d,%d/%d) #cs:%d, wins:%f count:%f, done:%t, turn:%d, #lc:%d",
		n.r, n.c, n.cellType, len(n.children), n.wins, n.count, n.done, n.turn, n.liveChildren)
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

func (root *mcNode) size() int {
	s := 0
	q := make([]*mcNode, 1, 1024)
	q[0] = root
	for len(q) > 0 {
		n := q[len(q)-1]
		q = q[:len(q)-1]
		s++
		q = append(q, n.children...)
	}
	return s
}

// Returns the number of leaf nodes on each depth level, starting from 0 for the root.
func (root *mcNode) leafNodesPerDepth() []int {
	c := []int{}
	type ni struct {
		n *mcNode
		d int
	}
	q := make([]ni, 1, 1024)
	q[0] = ni{root, 0}
	for len(q) > 0 {
		n := q[len(q)-1]
		q = q[:len(q)-1]
		if len(n.n.children) == 0 {
			if len(c) <= n.d {
				c1 := make([]int, n.d+1)
				copy(c1, c)
				c = c1
			}
			c[n.d]++
		}
		for _, c := range n.n.children {
			q = append(q, ni{c, n.d + 1})
		}
	}
	return c
}

type MCTS struct {
	rnd              *rand.Rand
	MaxFlagPositions int // maximum number of (random) positions to consider for placing a flag in a single move.
	UctFactor        float64
	FlagsFirst       bool // If true, flags will be played whenever possible.
	ReuseTree        bool
	root             *mcNode
	rootMove         int // Move number corresponding to the state at root.
}

func (mcts *MCTS) playRandomGame(ge SinglePlayerGameEngine, firstMove *mcNode) (winner int) {
	b := ge.Board()
	if !ge.MakeMove(GameEngineMove{
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
	if maxFlags <= 0 {
		maxFlags = len(b.FlatFields)
	}
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

func (mcts *MCTS) run(ge SinglePlayerGameEngine, path []*mcNode) (depth int) {
	node := path[len(path)-1]
	b := ge.Board()
	if node.children == nil {
		// Terminal node in our exploration graph, but not in the whole game:
		// While traversing a path we play moves and detect when the game IsDone (below).
		cs := mcts.nextMoves(node, b)
		if len(cs) == 0 {
			panic(fmt.Sprintf("No next moves on allegedly non-final node: %s", node.String()))
		}
		node.children = cs
		node.liveChildren = len(cs)
		// Play a random child (rollout)
		c := cs[mcts.rnd.Intn(len(cs))]
		winner := mcts.playRandomGame(ge, c)
		path = append(path, c)
		mcts.backpropagate(path, winner)
		return len(path)
	}
	// Node has children already, descend to the one with the highest UTC.
	c := mcts.getNextByUtc(node)
	if c == nil {
		// All children are done, but that was not properly propagated up to the parent node.
		panic(fmt.Sprintf("No children left for node: %s", node.String()))
	}
	move := GameEngineMove{
		playerNum: c.turn, move: b.Move, row: c.r, col: c.c, cellType: c.cellType,
	}
	if !ge.MakeMove(move) {
		panic(fmt.Sprintf("Failed to make move %s", move.String()))
	}
	path = append(path, c)
	if ge.IsDone() {
		// This was the last move. Propagate the result up.
		c.done = true
		winner := ge.Winner()
		mcts.backpropagate(path, winner)
		depth = len(path)
	} else {
		// Not done: descend to next level
		depth = mcts.run(ge, path)
	}
	if c.done {
		// Propagate up the fact that child is done to avoid revisiting it.
		node.liveChildren--
		if node.liveChildren == 0 {
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
	Iterations    int
	MaxDepth      int
	TreeSize      int
	LeafNodes     []int // Per depth level, 0=root
	Elapsed       time.Duration
	FullyExplored bool
	Moves         []MCTSMoveStats
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
	fmt.Fprintf(&sb, "N: %d\nmaxDepth:%d\nsize:%d\nelapsed:%.3f\nN/sec:%.1f\n",
		s.Iterations, s.MaxDepth, s.TreeSize, s.Elapsed.Seconds(), float64(s.Iterations)/s.Elapsed.Seconds())
	for _, m := range s.Moves {
		cellType := ""
		if m.cellType == cellFlag {
			cellType = " F"
		}
		fmt.Fprintf(&sb, "  (%d,%d%s) U:%.3f Q:%.2f N:%d\n", m.row, m.col, cellType, m.U, m.Q, m.iterations)
	}
	return sb.String()
}

func NewMCTS() *MCTS {
	return &MCTS{
		rnd:              rand.New(rand.NewSource(time.Now().UnixNano())),
		MaxFlagPositions: -1, // Unlimited
		UctFactor:        1.0,
		FlagsFirst:       false,
	}
}

func (mcts *MCTS) findBestMoveAndBuildStats(n *mcNode, elapsed time.Duration, maxDepth int, move int) (GameEngineMove, *MCTSStats) {

	// Return some stats
	stats := &MCTSStats{
		Iterations:    int(n.count),
		MaxDepth:      maxDepth,
		Elapsed:       elapsed,
		FullyExplored: n.done,
		TreeSize:      n.size(),
		LeafNodes:     n.leafNodesPerDepth(),
		Moves:         make([]MCTSMoveStats, len(n.children)),
	}
	var best *mcNode
	for i, c := range n.children {
		if best == nil || c.Q() > best.Q() {
			best = c
		}
		stats.Moves[i] = MCTSMoveStats{
			row:        c.r,
			col:        c.c,
			cellType:   c.cellType,
			iterations: int(c.count),
			U:          c.U(n.count, mcts.UctFactor),
			Q:          c.Q(),
		}
	}
	m := GameEngineMove{
		playerNum: best.turn,
		move:      move,
		row:       best.r,
		col:       best.c,
		cellType:  best.cellType,
	}
	return m, stats
}

func (mcts *MCTS) SuggestMove(gameEngine SinglePlayerGameEngine, maxDuration time.Duration) (GameEngineMove, *MCTSStats) {
	var root *mcNode
	moveHist := gameEngine.MoveHistory()
	if !mcts.ReuseTree || mcts.root == nil || len(moveHist) <= mcts.rootMove {
		// Start a new root.
		root = &mcNode{turn: gameEngine.Board().Turn}
	} else {
		// Try to find and resume a subtree for the current state of the board.
		r := mcts.root
		ok := true
		for i := 0; i < len(moveHist)-mcts.rootMove; i++ {
			found := false
			for _, c := range r.children {
				if c.r == moveHist[mcts.rootMove+i].row &&
					c.c == moveHist[mcts.rootMove+i].col &&
					c.turn == moveHist[mcts.rootMove+i].playerNum &&
					c.cellType == moveHist[mcts.rootMove+i].cellType {
					r = c
					found = true
					break
				}
			}
			if !found {
				ok = false
				break
			}
		}
		if ok {
			root = r
			fmt.Printf("Reusing tree of size %d\n", root.size())
		} else {
			root = &mcNode{turn: gameEngine.Board().Turn}
		}
	}
	mcts.root = root
	mcts.rootMove = gameEngine.Board().Move
	// Since we are reusing subtrees, we might already have fully explored
	// the subtree. In that case, pick the best child immediately
	if root.done {
		if len(root.children) == 0 {
			panic("No children, but root is done")
		}
		return mcts.findBestMoveAndBuildStats(root, time.Duration(0), 0, gameEngine.Board().Move)
	}
	started := time.Now()
	maxDepth := 0
	for n := 0; ; n++ {
		// Check every N rounds if we're done.
		if n&63 == 0 && time.Since(started) >= maxDuration {
			break
		}
		ge := gameEngine.Clone(mcts.rnd)
		path := make([]*mcNode, 1, 100)
		path[0] = root
		depth := mcts.run(ge, path)
		if depth > maxDepth {
			maxDepth = depth
		}
		if root.done {
			// Board completely explored
			break
		}
	}
	elapsed := time.Since(started)

	// Return some stats
	stats := &MCTSStats{
		Iterations:    int(root.count),
		MaxDepth:      maxDepth,
		Elapsed:       elapsed,
		FullyExplored: root.done,
		TreeSize:      root.size(),
		LeafNodes:     root.leafNodesPerDepth(),
		Moves:         make([]MCTSMoveStats, len(root.children)),
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
		playerNum: best.turn,
		move:      gameEngine.Board().Move,
		row:       best.r,
		col:       best.c,
		cellType:  best.cellType,
	}
	return move, stats
}
