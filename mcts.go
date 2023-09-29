package hexz

import (
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/dnswlt/hexz/xrand"
)

type mcNode struct {
	// bit-encoding of several values [i:j], j exclusive):
	// [ liveChildren[0:8], done[8], turn[9], cellType[10], r[16:24], c[24:32] ]
	wins     int
	count    int
	bits     uint32
	children []mcNode
}

func newMcNode(r, c int) mcNode {
	return mcNode{bits: uint32(r<<16) | (uint32(c) << 24)}
}

func (n *mcNode) r() int {
	return int((n.bits >> 16) & 0xff)
}

func (n *mcNode) c() int {
	return int((n.bits >> 24) & 0xff)
}

func (n *mcNode) done() bool {
	return n.bits&(1<<8) != 0
}

func (n *mcNode) setDone() {
	n.bits |= (1 << 8)
}

func (n *mcNode) turn() int {
	if n.bits&(1<<9) != 0 {
		return 2
	}
	return 1
}

func (n *mcNode) setTurn(turn int) {
	if turn == 2 {
		n.bits |= 1 << 9
		return
	}
	n.bits &= ^uint32(1 << 9)
}

func (n *mcNode) cellType() CellType {
	if n.bits&(1<<10) != 0 {
		return cellFlag
	}
	return cellNormal
}

func (n *mcNode) setFlag() {
	n.bits |= 1 << 10
}

func (n *mcNode) liveChildren() int {
	return int(n.bits & 0x7f)
}

func (n *mcNode) setLiveChildren(k int) {
	if k > 127 {
		panic(fmt.Sprintf("setLiveChildren called with large k: %d", k))
	}
	n.bits = (n.bits & ^uint32(0x7f)) | uint32(k)
}

func (n *mcNode) decrLiveChildren() {
	n.bits--
}

func (n *mcNode) String() string {
	return fmt.Sprintf("(%d,%d/%d) #cs:%d, wins:%d count:%d, done:%t, turn:%d, #lc:%d",
		n.r(), n.c(), n.cellType(), len(n.children), n.wins, n.count, n.done(), n.turn(), n.liveChildren())
}

func (n *mcNode) Q() float64 {
	if n.count == 0 {
		return 0
	}
	return float64(n.wins) / float64(n.count)
}

// Tabulating logs showed a significant performance gain on amd64.
//
// go test -bench BenchmarkMCTSRun -run ^$ -count=10
//
// dnswlt/hexz$ benchstat old.txt new.txt
// goos: darwin
// goarch: arm64
// pkg: github.com/dnswlt/hexz
//
//	│   old.txt   │              new.txt               │
//	│   sec/op    │   sec/op     vs base               │
//
// MCTSRun-10   25.17µ ± 2%   23.52µ ± 1%  -6.55% (p=0.000 n=10)
const useTabulatedLogs = true
const logValuesSize = 100_000

var logValues [logValuesSize]float64

func init() {
	for i := 1; i < logValuesSize; i++ {
		logValues[i] = math.Log(float64(i))
	}
}

var EnableInitialDrawAssumption = true

func (n *mcNode) U(parentCount int, uctFactor float64) float64 {
	if !EnableInitialDrawAssumption && n.count == 0 {
		return math.MaxFloat64
	}
	if parentCount == 0 {
		// First rollout for the parent. Assume this game was a draw.
		return 0.5
	}
	var l float64
	if useTabulatedLogs && parentCount < logValuesSize {
		l = logValues[parentCount]
	} else {
		l = math.Log(float64(parentCount))
	}
	if n.count == 0 {
		// Never played => assume one game was played and it was a draw.
		return 0.5 + uctFactor*math.Sqrt(l)
	}
	return float64(n.wins)/float64(n.count) + uctFactor*math.Sqrt(l/float64(n.count))
}

// Returns the number of leaf and branch nodes on each depth level, starting from 0 for the root.
// The depth of the tree can be computed as len(leafNodes).
func (root *mcNode) nodesPerDepth() (size int, leafNodes []int, branchNodes []int, visitCounts []map[int]int) {
	type ni struct {
		n *mcNode
		d int
	}
	q := make([]ni, 1, 1024)
	q[0] = ni{root, 0}
	for len(q) > 0 {
		n := q[len(q)-1]
		q = q[:len(q)-1]
		size++
		if len(leafNodes) == n.d {
			leafNodes = append(leafNodes, 0)
			branchNodes = append(branchNodes, 0)
			visitCounts = append(visitCounts, make(map[int]int))
		}
		if len(n.n.children) == 0 {
			leafNodes[n.d]++
		} else {
			branchNodes[n.d]++
		}
		visitCounts[n.d][int(n.n.count)]++
		for i := range n.n.children {
			q = append(q, ni{&n.n.children[i], n.d + 1})
		}
	}
	return
}

type MCTS struct {
	UctFactor float64
	Mem       []mcNode
	Next      int
}

func (mcts *MCTS) playRandomGame(ge *GameEngineFlagz, firstMove *mcNode) (winner int) {
	b := ge.Board()
	if !ge.MakeMove(GameEngineMove{
		PlayerNum: firstMove.turn(),
		Move:      b.Move,
		Row:       firstMove.r(),
		Col:       firstMove.c(),
		CellType:  firstMove.cellType(),
	}) {
		panic("Invalid move: " + firstMove.String())
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
	for i := range node.children {
		l := &node.children[i]
		if l.done() {
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

func (mcts *MCTS) nextMoves(ge *GameEngineFlagz) []mcNode {
	b := ge.B
	hasFlag := b.Resources[b.Turn-1].NumPieces[cellFlag] > 0
	nChildren := ge.NormalMoves[b.Turn-1]
	if hasFlag {
		nChildren += ge.FreeCells
	}
	var cs []mcNode
	if mcts.Mem != nil {
		cs = mcts.Mem[mcts.Next : mcts.Next+nChildren]
		mcts.Next += nChildren
	} else {
		cs = make([]mcNode, nChildren)
	}
	i := 0
	for r := 0; r < len(b.Fields); r++ {
		for c := 0; c < len(b.Fields[r]); c++ {
			f := &b.Fields[r][c]
			if f.occupied() {
				continue
			}
			if f.isAvail(b.Turn) {
				cs[i].bits = uint32(r)<<16 | uint32(c)<<24
				cs[i].setTurn(b.Turn)
				i++
			}
			if hasFlag {
				cs[i].bits = uint32(r)<<16 | uint32(c)<<24
				cs[i].setTurn(b.Turn)
				cs[i].setFlag()
				i++
			}
		}
	}
	if len(cs) != i {
		// If this happens, our memory allocation above is wrong.
		panic(fmt.Sprintf("Wrong number of next moves: %d != %d+%d", len(cs), ge.FreeCells, ge.NormalMoves[b.Turn-1]))
	}
	return cs
}

func (mcts *MCTS) run(ge *GameEngineFlagz, node *mcNode, curDepth int) (winner int, depth int) {
	b := ge.Board()
	if node.children == nil {
		// Terminal node in our exploration graph, but not in the whole game:
		// While traversing a path we play moves and detect when the game IsDone (below).
		cs := mcts.nextMoves(ge)
		if len(cs) == 0 {
			panic(fmt.Sprintf("No next moves on allegedly non-final node: %s", node.String()))
		}
		node.children = cs
		node.setLiveChildren(len(cs))
		// Play a random child (rollout)
		c := &cs[xrand.Intn(len(cs))]
		winner = mcts.playRandomGame(ge, c)
		// Record counts and wins for both nodes.
		node.count++
		c.count++
		if winner == c.turn() {
			c.wins++
		}
		if winner == node.turn() {
			node.wins++
		}
		return winner, curDepth
	}
	// Node has children already, descend to the one with the highest UTC.
	c := mcts.getNextByUtc(node)
	if c == nil {
		// All children are done, but that was not properly propagated up to the parent node.
		panic(fmt.Sprintf("No children left for node: %s", node.String()))
	}
	move := GameEngineMove{
		PlayerNum: c.turn(), Move: b.Move, Row: c.r(), Col: c.c(), CellType: c.cellType(),
	}
	if !ge.MakeMove(move) {
		panic(fmt.Sprintf("Failed to make move %s", move.String()))
	}
	if ge.IsDone() {
		// This was the last move. Propagate the result up.
		c.setDone()
		winner, depth = ge.Winner(), curDepth
	} else {
		// Not done: descend to next level
		winner, depth = mcts.run(ge, c, curDepth+1)
	}
	if c.done() {
		// Propagate up the fact that child is done to avoid revisiting it.
		node.decrLiveChildren()
		if node.liveChildren() == 0 {
			node.setDone()
		}
	}
	node.count++
	if winner == node.turn() {
		node.wins++
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
	LeafNodes     []int         // Per depth level, 0=root
	BranchNodes   []int         // Per depth level, 0=root
	VisitCounts   []map[int]int // Per depth level, maps visit count to number of nodes with that count.
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
		UctFactor: 1.0,
	}
}

func NewMCTSWithMem(cap int) *MCTS {
	return &MCTS{
		UctFactor: 1.0,
		Mem:       make([]mcNode, cap),
		Next:      0,
	}
}

func (mcts *MCTS) bestNextMoveWithStats(root *mcNode, elapsed time.Duration, move int) (GameEngineMove, *MCTSStats) {
	size, leafNodes, branchNodes, visitCounts := root.nodesPerDepth()
	stats := &MCTSStats{
		Iterations:    int(root.count),
		MaxDepth:      len(leafNodes),
		Elapsed:       elapsed,
		FullyExplored: root.done(),
		TreeSize:      size,
		LeafNodes:     leafNodes,
		BranchNodes:   branchNodes,
		VisitCounts:   visitCounts,
		Moves:         make([]MCTSMoveStats, len(root.children)),
	}
	best := &root.children[0]
	for i := range root.children {
		c := &root.children[i]
		if c.Q() > best.Q() {
			best = c
		}
		stats.Moves[i] = MCTSMoveStats{
			Row:        c.r(),
			Col:        c.c(),
			CellType:   c.cellType(),
			Iterations: int(c.count),
			U:          float64(c.U(root.count, mcts.UctFactor)),
			Q:          float64(c.Q()),
		}
	}
	m := GameEngineMove{
		PlayerNum: best.turn(),
		Move:      move,
		Row:       best.r(),
		Col:       best.c(),
		CellType:  best.cellType(),
	}
	return m, stats
}

func (mcts *MCTS) Reset() {
	if mcts.Mem != nil {
		mcts.Next = 0
	}
}

func (mcts *MCTS) SuggestMove(gameEngine *GameEngineFlagz, maxDuration time.Duration) (GameEngineMove, *MCTSStats) {
	defer mcts.Reset()
	root := &mcNode{}
	root.setTurn(gameEngine.Board().Turn)
	started := time.Now()
	ge := gameEngine.Clone()
	for n := 0; ; n++ {
		// Check every N rounds if we're done. Run at least once.
		if (n-1)&63 == 0 && time.Since(started) >= maxDuration {
			break
		}
		ge.copyFrom(gameEngine)
		mcts.run(ge, root, 0)
		if root.done() {
			// Board completely explored
			break
		}
	}
	elapsed := time.Since(started)
	return mcts.bestNextMoveWithStats(root, elapsed, gameEngine.Board().Move)
}
