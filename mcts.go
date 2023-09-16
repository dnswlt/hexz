package hexz

import (
	"fmt"
	"log"
	"math"
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
		// Never played => infinitely interesting.
		return math.MaxFloat64
	}
	return n.wins/n.count + uctFactor*math.Sqrt(math.Log(parentCount)/n.count)
}

// Returns the number of leaf and branch nodes on each depth level, starting from 0 for the root.
func (root *mcNode) nodesPerDepth() (size int, leafNodes []int, branchNodes []int) {
	ls := []int{}
	bs := []int{}
	s := 0
	type ni struct {
		n *mcNode
		d int
	}
	q := make([]ni, 1, 1024)
	q[0] = ni{root, 0}
	for len(q) > 0 {
		n := q[len(q)-1]
		q = q[:len(q)-1]
		s++
		if len(ls) <= n.d {
			ls1, bs1 := make([]int, n.d+1), make([]int, n.d+1)
			copy(ls1, ls)
			copy(bs1, bs)
			ls, bs = ls1, bs1
		}
		if len(n.n.children) == 0 {
			ls[n.d]++
		} else {
			bs[n.d]++
		}
		for _, c := range n.n.children {
			q = append(q, ni{c, n.d + 1})
		}
	}
	return s, ls, bs
}

type MCTS struct {
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
		PlayerNum: firstMove.turn,
		Move:      b.Move,
		Row:       firstMove.r,
		Col:       firstMove.c,
		CellType:  firstMove.cellType,
	}) {
		panic("Invalid move")
	}
	for !ge.IsDone() {
		m, err := ge.RandomMove()
		if err != nil {
			log.Fatalf("Could not suggest a move: %s", err.Error())
		}
		if !ge.MakeMove(*m) {
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
			if hasFlag && (randFloat64() < float64(maxFlags)/(float64(nFlags)+1)) {
				// reservoir sampling to pick maxFlags with equal probability among all possibilities.
				k := nFlags
				if k >= maxFlags {
					k = randIntn(maxFlags)
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
		c := cs[randIntn(len(cs))]
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
		PlayerNum: c.turn, Move: b.Move, Row: c.r, Col: c.c, CellType: c.cellType,
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
	Row        int
	Col        int
	CellType   CellType
	U          float64
	Q          float64
	Iterations int
}

type MCTSStats struct {
	Iterations    int
	MaxDepth      int
	TreeSize      int
	LeafNodes     []int // Per depth level, 0=root
	BranchNodes   []int // Per depth level, 0=root
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

func (s *MCTSStats) MoveScores() *MoveScores {
	normalCell := make([][]float64, numBoardRows)
	flag := make([][]float64, numBoardRows)
	for i := 0; i < numBoardRows; i++ {
		nCols := numFieldsFirstRow - i%2
		normalCell[i] = make([]float64, nCols)
		flag[i] = make([]float64, nCols)
	}
	for _, m := range s.Moves {
		switch m.CellType {
		case cellNormal:
			normalCell[m.Row][m.Col] = m.Q
		case cellFlag:
			flag[m.Row][m.Col] = m.Q
		}
	}
	return &MoveScores{
		NormalCell: normalCell,
		Flag:       flag,
	}
}

func (s *MCTSStats) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "N: %d\nmaxDepth:%d\nsize:%d\nelapsed:%.3f\nN/sec:%.1f\n",
		s.Iterations, s.MaxDepth, s.TreeSize, s.Elapsed.Seconds(), float64(s.Iterations)/s.Elapsed.Seconds())
	for _, m := range s.Moves {
		cellType := ""
		if m.CellType == cellFlag {
			cellType = " F"
		}
		fmt.Fprintf(&sb, "  (%d,%d%s) U:%.3f Q:%.2f N:%d\n", m.Row, m.Col, cellType, m.U, m.Q, m.Iterations)
	}
	return sb.String()
}

func NewMCTS() *MCTS {
	return &MCTS{
		MaxFlagPositions: -1, // Unlimited
		UctFactor:        1.0,
		FlagsFirst:       false,
		ReuseTree:        true,
	}
}

func (mcts *MCTS) bestNextMoveWithStats(n *mcNode, elapsed time.Duration, maxDepth int, move int) (GameEngineMove, *MCTSStats) {
	size, leafNodes, branchNodes := mcts.root.nodesPerDepth()
	stats := &MCTSStats{
		Iterations:    int(n.count),
		MaxDepth:      maxDepth,
		Elapsed:       elapsed,
		FullyExplored: n.done,
		TreeSize:      size,
		LeafNodes:     leafNodes,
		BranchNodes:   branchNodes,
		Moves:         make([]MCTSMoveStats, len(n.children)),
	}
	var best *mcNode
	for i, c := range n.children {
		if best == nil || c.Q() > best.Q() {
			best = c
		}
		stats.Moves[i] = MCTSMoveStats{
			Row:        c.r,
			Col:        c.c,
			CellType:   c.cellType,
			Iterations: int(c.count),
			U:          c.U(n.count, mcts.UctFactor),
			Q:          c.Q(),
		}
	}
	m := GameEngineMove{
		PlayerNum: best.turn,
		Move:      move,
		Row:       best.r,
		Col:       best.c,
		CellType:  best.cellType,
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
				if c.r == moveHist[mcts.rootMove+i].Row &&
					c.c == moveHist[mcts.rootMove+i].Col &&
					c.turn == moveHist[mcts.rootMove+i].PlayerNum &&
					c.cellType == moveHist[mcts.rootMove+i].CellType {
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
		} else {
			root = &mcNode{turn: gameEngine.Board().Turn}
		}
	}
	if mcts.ReuseTree {
		mcts.root = root
		mcts.rootMove = gameEngine.Board().Move
	}
	// If we are reusing subtrees, we might already have fully explored
	// the subtree. In that case, pick the best child immediately
	if root.done {
		if len(root.children) == 0 {
			panic("No children, but root is done")
		}
		return mcts.bestNextMoveWithStats(root, time.Duration(0), 0, gameEngine.Board().Move)
	}
	started := time.Now()
	maxDepth := 0
	for n := 0; ; n++ {
		// Check every N rounds if we're done. Run at least once.
		if (n-1)&63 == 0 && time.Since(started) >= maxDuration {
			break
		}
		ge := gameEngine.Clone()
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
	return mcts.bestNextMoveWithStats(root, elapsed, maxDepth, gameEngine.Board().Move)
}
