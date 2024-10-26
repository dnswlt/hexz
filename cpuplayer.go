package hexz

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/dnswlt/hexz/hexzpb"
	pb "github.com/dnswlt/hexz/hexzpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type CPUPlayer interface {
	SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error)
	MaxThinkTime() time.Duration
}

type LocalCPUPlayer struct {
	playerId PlayerId
	// Channel over which moves are sent
	mcts         *MCTS
	thinkTime    time.Duration // Current think time (auto-adjusted based on confidence)
	maxThinkTime time.Duration // Maximum think time. Upper bound, independent of the context's deadline.
}

func NewLocalCPUPlayer(playerId PlayerId, maxThinkTime time.Duration) *LocalCPUPlayer {
	return &LocalCPUPlayer{
		playerId:     playerId,
		mcts:         NewMCTS(),
		thinkTime:    maxThinkTime,
		maxThinkTime: maxThinkTime,
	}
}

func (cpu *LocalCPUPlayer) MaxThinkTime() time.Duration {
	return cpu.maxThinkTime
}

// SuggestMove calculates a suggested move (using MCTS).
// The GameEngineFlagz ge will not be modified.
func (cpu *LocalCPUPlayer) SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error) {
	t := cpu.thinkTime
	if t > cpu.maxThinkTime {
		t = cpu.maxThinkTime
	}
	mv, stats := cpu.mcts.SuggestMove(ge, t)
	if minQ := stats.MinQ(); minQ >= 0.98 || minQ <= 0.02 {
		// Speed up if we think we (almost) won or lost, but stop at 0.1% of maxThinkTime.
		if t > cpu.maxThinkTime/500 {
			cpu.thinkTime = t / 2
		}
	} else {
		cpu.thinkTime = cpu.maxThinkTime // use full time allowed.
	}
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
	return &GameEngineMove{
			PlayerNum: ge.B.Turn,
			Move:      mv.Move,
			Row:       mv.Row,
			Col:       mv.Col,
			CellType:  mv.CellType,
		}, &pb.SuggestMoveStats{
			Moves: moveEvals,
			Value: float32(2*stats.BestMoveQ - 1), // Normalize to the [-1..1] range returned by neural players.
		}, nil
}

type RemoteCPUPlayer struct {
	playerId     PlayerId
	addr         string // Address of the remote CPU player server, e.g. "localhost:50051"
	maxThinkTime time.Duration
	client       hexzpb.CPUPlayerServiceClient
}

func NewRemoteCPUPlayer(playerId PlayerId, addr string, maxThinkTime time.Duration) (*RemoteCPUPlayer, error) {
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}
	conn, err := grpc.NewClient(addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("cannot create gRPC client: %w", err)
	}
	client := hexzpb.NewCPUPlayerServiceClient(conn)
	return &RemoteCPUPlayer{
		playerId:     playerId,
		addr:         addr,
		maxThinkTime: maxThinkTime,
		client:       client,
	}, nil
}

func (cpu *RemoteCPUPlayer) MaxThinkTime() time.Duration {
	return cpu.maxThinkTime
}

func (cpu *RemoteCPUPlayer) SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error) {
	state, err := ge.Encode()
	if err != nil {
		return nil, nil, fmt.Errorf("cannot encode GameEngineFlagz: %v", err)
	}
	req := &pb.SuggestMoveRequest{
		MaxThinkTimeMs:  cpu.maxThinkTime.Milliseconds(),
		GameEngineState: state,
	}
	// Allow the RPC at most the think time, plus some buffer.
	ctx, cancel := context.WithTimeout(ctx, cpu.maxThinkTime+time.Duration(1)*time.Second)
	defer cancel()
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
	// Needs to be embedded to have MoveSuggesterServer implement hexzpb.CPUPlayerServiceServer.
	hexzpb.UnimplementedCPUPlayerServiceServer
	config *MoveSuggesterServerConfig
}

func NewMoveSuggesterServer(config *MoveSuggesterServerConfig) *MoveSuggesterServer {
	return &MoveSuggesterServer{
		config: config,
	}
}

func (s *MoveSuggesterServer) SuggestMove(ctx context.Context, req *hexzpb.SuggestMoveRequest) (*hexzpb.SuggestMoveResponse, error) {
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
	thinkTime := time.Duration(req.MaxThinkTimeMs) * time.Millisecond
	if s.config.CpuThinkTime > 0 && thinkTime > s.config.CpuThinkTime {
		thinkTime = s.config.CpuThinkTime
	}
	mv, _ := mcts.SuggestMove(ge, thinkTime)
	return &hexzpb.SuggestMoveResponse{
		Move: mv.Proto(),
		// TODO: add stats
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
	hexzpb.RegisterCPUPlayerServiceServer(grpcServer, s)
	return grpcServer.Serve(lis)
}
