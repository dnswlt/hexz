package hexz

import (
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/dnswlt/hexz/xrand"
)

// Nodes of the MCTS search tree.
type mcNode struct {
	wins  int32
	count int32
	// bit-encoding of several values [i:j], j exclusive):
	// [ r[0:8], c[8:16], turn[16], cellType[17:21]  ]
	bits     uint32
	children []mcNode
}

func (n *mcNode) set(r, c int, turn int, cellType CellType) {
	n.bits = uint32(r) | (uint32(c) << 8) | (uint32(turn>>1) << 16) | (uint32(cellType) << 17)
}

func (n *mcNode) incr(winner int) {
	n.count++
	if winner == n.turn() {
		n.wins++
	} else if winner != 0 {
		// Other player won.
		n.wins--
	}
	// Do nothing on a draw (count it as 0).
}

func (n *mcNode) r() int {
	return int(n.bits & 0xff)
}

func (n *mcNode) c() int {
	return int((n.bits >> 8) & 0xff)
}

func (n *mcNode) turn() int {
	if n.bits&(1<<16) != 0 {
		return 2
	}
	return 1
}

func (n *mcNode) cellType() CellType {
	return CellType((n.bits >> 17) & 0xf)
}

func (n *mcNode) String() string {
	return fmt.Sprintf("(%d,%d/%d) #cs:%d, wins:%d count:%d, turn:%d",
		n.r(), n.c(), n.cellType(), len(n.children), n.wins, n.count, n.turn())
}

func (n *mcNode) Q() float64 {
	if n.count == 0 {
		return 0
	}
	return 0.5 + float64(n.wins)/float64(n.count)/2
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

func (n *mcNode) U(parentCount int32, uctFactor float64) float64 {
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
	return n.Q() + uctFactor*math.Sqrt(l/float64(n.count))
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
	// If true, SuggestMove returns the most frequently vistited child node,
	// not the one with the highest win rate.
	ReturnMostFrequentlyVisited bool
	// Sample boards that led to a win/loss.
	WinningBoard *Board
	LosingBoard  *Board
	// For explicit memory handling.
	Mem  []mcNode
	Next int
}

func (mcts *MCTS) playRandomGame(ge *GameEngineFlagz) (winner int) {
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
				cs[i].set(r, c, b.Turn, cellNormal)
				i++
			}
			if hasFlag {
				cs[i].set(r, c, b.Turn, cellFlag)
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

func (mcts *MCTS) run(ge *GameEngineFlagz, node *mcNode) (winner int) {
	if ge.IsDone() {
		panic("Called Run() on a finished game")
	}
	return mcts.runInternal(ge, node, 0)
}

func (mcts *MCTS) runInternal(ge *GameEngineFlagz, node *mcNode, curDepth int) (winner int) {
	b := ge.Board()
	if node.children == nil {
		// Terminal node in our exploration graph, but not in the whole game: rollout time!
		cs := mcts.nextMoves(ge)
		if len(cs) == 0 {
			panic("No next moves, but game is not over yet")
		}
		node.children = cs
		// Play a random child (rollout)
		c := &cs[xrand.Intn(len(cs))]
		move := GameEngineMove{
			PlayerNum: c.turn(), Move: b.Move, Row: c.r(), Col: c.c(), CellType: c.cellType(),
		}
		if !ge.MakeMove(move) {
			panic(fmt.Sprintf("Failed to make move for rollout: %s", move.String()))
		}
		winner = mcts.playRandomGame(ge)
		if winner == 1 && mcts.WinningBoard == nil {
			mcts.WinningBoard = b.Copy()
		} else if winner == 2 && mcts.LosingBoard == nil {
			mcts.LosingBoard = b.Copy()
		}
		// Record counts and wins for child.
		c.incr(winner)
	} else {
		// Node has children already, descend to the one with the highest UTC.
		c := mcts.getNextByUtc(node)
		move := GameEngineMove{
			PlayerNum: c.turn(), Move: b.Move, Row: c.r(), Col: c.c(), CellType: c.cellType(),
		}
		if !ge.MakeMove(move) {
			panic(fmt.Sprintf("Failed to make move during tree descent: %s", move.String()))
		}
		if ge.IsDone() {
			// This was the last move. Update child stats.
			winner = ge.Winner()
			c.incr(winner)
		} else {
			// Not done: descend to next level
			winner = mcts.runInternal(ge, c, curDepth+1)
		}
	}
	node.incr(winner)
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
	Iterations  int
	MaxDepth    int
	TreeSize    int
	LeafNodes   []int         // Per depth level, 0=root
	BranchNodes []int         // Per depth level, 0=root
	VisitCounts []map[int]int // Per depth level, maps visit count to number of nodes with that count.
	Elapsed     time.Duration
	Moves       []MCTSMoveStats
	BestMoveQ   float64
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
		// Returning the node with the highest number of visits is the "standard approach",
		// https://ai.stackexchange.com/questions/16905/mcts-how-to-choose-the-final-action-from-the-root
		ReturnMostFrequentlyVisited: true,
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
	moves := make([]MCTSMoveStats, len(root.children))
	best := &root.children[0]
	for i := range root.children {
		c := &root.children[i]
		if !mcts.ReturnMostFrequentlyVisited && c.Q() > best.Q() {
			best = c
		} else if mcts.ReturnMostFrequentlyVisited && c.count > best.count {
			best = c
		}
		moves[i] = MCTSMoveStats{
			Row:        c.r(),
			Col:        c.c(),
			CellType:   c.cellType(),
			Iterations: int(c.count),
			U:          c.U(root.count, mcts.UctFactor),
			Q:          c.Q(),
		}
	}
	stats := &MCTSStats{
		Iterations:  int(root.count),
		MaxDepth:    len(leafNodes),
		Elapsed:     elapsed,
		TreeSize:    size,
		LeafNodes:   leafNodes,
		BranchNodes: branchNodes,
		VisitCounts: visitCounts,
		Moves:       moves,
		BestMoveQ:   best.Q(),
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
	mcts.WinningBoard = nil
	mcts.LosingBoard = nil
}

func (mcts *MCTS) SuggestMove(gameEngine *GameEngineFlagz, maxDuration time.Duration, maxIterations int) (GameEngineMove, *MCTSStats) {
	mcts.Reset()
	root := &mcNode{}
	root.set(0, 0, gameEngine.Board().Turn, cellNormal) // Dummy values, only the turn matters.
	started := time.Now()
	ge := gameEngine.Clone()
	if maxIterations <= 0 {
		if maxDuration > 0 {
			maxIterations = math.MaxInt // Only limited by duration.
		} else {
			maxIterations = 1 // Run at least once.
		}
	}
	for n := 0; n < maxIterations; n++ {
		// Only check every N rounds if we're done to avoid excessive clock reads.
		// Run at least once.
		if maxDuration > 0 && (n-1)&63 == 0 && time.Since(started) >= maxDuration {
			break
		}
		ge.copyFrom(gameEngine)
		mcts.run(ge, root)
	}
	elapsed := time.Since(started)
	return mcts.bestNextMoveWithStats(root, elapsed, gameEngine.Board().Move)
}
