package hexz

import (
	"context"
	"fmt"
	"net"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type CPUPlayer interface {
	SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error)
}

type LocalCPUPlayer struct {
	playerId PlayerId
	// Channel over which moves are sent
	mcts          *MCTS
	thinkTime     time.Duration // Current think time (auto-adjusted based on confidence)
	maxThinkTime  time.Duration // Maximum think time. Upper bound, independent of the context's deadline.
	maxIterations int           // Maximum number of MCTS iterations per move. <= 0 means unbounded.
}

func NewLocalCPUPlayer(playerId PlayerId, maxThinkTime time.Duration, maxIterations int) *LocalCPUPlayer {
	return &LocalCPUPlayer{
		playerId:      playerId,
		mcts:          NewMCTS(),
		thinkTime:     maxThinkTime,
		maxThinkTime:  maxThinkTime,
		maxIterations: maxIterations,
	}
}

func MCTSStatsToProto(stats *MCTSStats) *pb.SuggestMoveStats {
	moveEvals := make([]*pb.SuggestMoveStats_ScoredMove, len(stats.Moves))
	for i, m := range stats.Moves {
		moveEvals[i] = &pb.SuggestMoveStats_ScoredMove{
			Row:  int32(m.Row),
			Col:  int32(m.Col),
			Type: pb.Field_CellType(m.CellType),
			Scores: []*pb.SuggestMoveStats_Score{
				{Kind: pb.SuggestMoveStats_FINAL, Score: float32(m.Iterations) / float32(stats.Iterations)},
			},
		}
	}
	return &pb.SuggestMoveStats{
		Moves: moveEvals,
		Value: float32(2*stats.BestMoveQ - 1), // Normalize to the [-1..1] range returned by neural players.
	}
}

// SuggestMove calculates a suggested move (using MCTS).
// The GameEngineFlagz ge will not be modified.
func (cpu *LocalCPUPlayer) SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error) {
	t := cpu.thinkTime
	if t > cpu.maxThinkTime {
		t = cpu.maxThinkTime
	}
	mv, stats := cpu.mcts.SuggestMove(ge, t, cpu.maxIterations)
	if minQ := stats.MinQ(); minQ >= 0.98 || minQ <= 0.02 {
		// Speed up if we think we (almost) won or lost, but stop at 0.1% of maxThinkTime.
		if t > cpu.maxThinkTime/500 {
			cpu.thinkTime = t / 2
		}
	} else {
		cpu.thinkTime = cpu.maxThinkTime // use full time allowed.
	}
	return &GameEngineMove{
		PlayerNum: ge.B.Turn,
		Move:      mv.Move,
		Row:       mv.Row,
		Col:       mv.Col,
		CellType:  mv.CellType,
	}, MCTSStatsToProto(stats), nil
}

type RemoteCPUPlayer struct {
	playerId      PlayerId
	addr          string // Address of the remote CPU player server, e.g. "localhost:50051"
	maxThinkTime  time.Duration
	maxIterations int
	client        pb.CPUPlayerServiceClient
}

func NewRemoteCPUPlayer(playerId PlayerId, addr string, maxThinkTime time.Duration, maxIterations int) (*RemoteCPUPlayer, error) {
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}
	conn, err := grpc.NewClient(addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("cannot create gRPC client: %w", err)
	}
	client := pb.NewCPUPlayerServiceClient(conn)
	return &RemoteCPUPlayer{
		playerId:      playerId,
		addr:          addr,
		maxThinkTime:  maxThinkTime,
		maxIterations: maxIterations,
		client:        client,
	}, nil
}

func (cpu *RemoteCPUPlayer) SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error) {
	state, err := ge.Encode()
	if err != nil {
		return nil, nil, fmt.Errorf("cannot encode GameEngineFlagz: %v", err)
	}
	req := &pb.SuggestMoveRequest{
		MaxThinkTimeMs:  cpu.maxThinkTime.Milliseconds(),
		MaxIterations:   int64(cpu.maxIterations),
		GameEngineState: state,
	}
	// Allow the RPC at most the think time, plus some buffer. If think time is unbounded, don't set a deadline.
	if cpu.maxThinkTime > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cpu.maxThinkTime+time.Duration(1)*time.Second)
		defer cancel()
	}
	resp, err := cpu.client.SuggestMove(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC SuggestMove request failed: %v", err)
	}
	if resp.Move == nil {
		return nil, nil, fmt.Errorf("SuggestMoveResponse contains no move")
	}
	geMove := &GameEngineMove{}
	geMove.DecodeProto(resp.Move)
	return geMove, resp.MoveStats, nil
}

type MoveSuggesterServerConfig struct {
	Addr         string // e.g. "localhost:50051".
	CpuThinkTime time.Duration
	CpuMaxFlags  int
}

type MoveSuggesterServer struct {
	// Needs to be embedded to have MoveSuggesterServer implement pb.CPUPlayerServiceServer.
	pb.UnimplementedCPUPlayerServiceServer
	config *MoveSuggesterServerConfig
}

func NewMoveSuggesterServer(config *MoveSuggesterServerConfig) *MoveSuggesterServer {
	return &MoveSuggesterServer{
		config: config,
	}
}

func (s *MoveSuggesterServer) SuggestMove(ctx context.Context, req *pb.SuggestMoveRequest) (*pb.SuggestMoveResponse, error) {
	if req.GameEngineState == nil {
		return nil, fmt.Errorf("missing game engine")
	}
	var ge *GameEngineFlagz
	switch state := req.GameEngineState.State.(type) {
	case *pb.GameEngineState_Flagz:
		ge = NewGameEngineFlagz()
		if err := ge.Decode(req.GameEngineState); err != nil {
			return nil, fmt.Errorf("invalid game engine state")
		}
	default:
		return nil, fmt.Errorf("unsupported game type: %T", state)
	}
	mcts := NewMCTS()

	if req.MaxThinkTimeMs <= 0 && req.MaxIterations <= 0 && s.config.CpuThinkTime <= 0 {
		return nil, fmt.Errorf("neither max_think_time_ms nor max_iterations specified")
	}
	thinkTime := time.Duration(req.MaxThinkTimeMs) * time.Millisecond
	// Always respect the maximum think time set by this server.
	if s.config.CpuThinkTime > 0 && (thinkTime <= 0 || thinkTime > s.config.CpuThinkTime) {
		thinkTime = s.config.CpuThinkTime
	}
	mv, stats := mcts.SuggestMove(ge, thinkTime, int(req.MaxIterations))
	return &pb.SuggestMoveResponse{
		Move:      mv.Proto(),
		MoveStats: MCTSStatsToProto(stats),
	}, nil
}

func (s *MoveSuggesterServer) Serve() error {
	lis, err := net.Listen("tcp", s.config.Addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}
	infoLog.Printf("MoveSuggesterServer listening on %s", s.config.Addr)
	var opts []grpc.ServerOption
	grpcServer := grpc.NewServer(opts...)
	pb.RegisterCPUPlayerServiceServer(grpcServer, s)
	return grpcServer.Serve(lis)
}
