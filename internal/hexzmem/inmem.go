package hexzmem

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

// When run as a standalone app, we can store all logged in users.
// When running as a container, we can't do that, so we need to
// store logins in some external storage (Redis).

const (
	maxLoggedInPlayers = 10000
)

var (
	errPlayerNotFound = errors.New("player not found")
)

type InMemoryPlayerStore struct {
	// Contains all logged in players, mapped by their (cookie) playerId.
	players     map[api.PlayerId]*api.Player
	mut         sync.Mutex
	loginTTL    time.Duration // How long a login is valid.
	lastCleanup time.Time
}

// Creates a new in-memory player store and loads the player DB from the given file.
// If dbPath is empty, no persistent storage is used.
func NewInMemoryPlayerStore(loginTTL time.Duration) (*InMemoryPlayerStore, error) {
	s := &InMemoryPlayerStore{
		players:  make(map[api.PlayerId]*api.Player),
		loginTTL: loginTTL,
	}
	return s, nil
}

func (s *InMemoryPlayerStore) Lookup(ctx context.Context, playerId api.PlayerId) (api.Player, error) {
	s.mut.Lock()
	defer s.mut.Unlock()
	// Clean up periodically.
	if time.Since(s.lastCleanup) > 1*time.Minute {
		now := time.Now()
		for pId, p := range s.players {
			if now.Sub(p.LastActive) > s.loginTTL {
				delete(s.players, pId)
			}
		}
		s.lastCleanup = now
	}
	p, ok := s.players[playerId]
	if !ok {
		return api.Player{}, errPlayerNotFound
	}
	p.LastActive = time.Now()
	return *p, nil
}

func (s *InMemoryPlayerStore) Login(ctx context.Context, playerId api.PlayerId, name string) error {
	s.mut.Lock()
	defer s.mut.Unlock()
	if len(s.players) >= maxLoggedInPlayers {
		return fmt.Errorf("too many logged in players")
	}
	s.players[playerId] = &api.Player{
		Id:         playerId,
		Name:       name,
		LastActive: time.Now(),
	}
	return nil
}

func (s *InMemoryPlayerStore) Logout(ctx context.Context, playerId api.PlayerId) error {
	s.mut.Lock()
	defer s.mut.Unlock()
	delete(s.players, playerId)
	return nil
}

func (s *InMemoryPlayerStore) NumPlayers() int {
	s.mut.Lock()
	defer s.mut.Unlock()
	return len(s.players)
}

var (
	ErrGameNotExist = fmt.Errorf("game does not exist in store")
)

// An in-memory replacement for a RedisClient, to enable purely local, single-server
// gameplay.
type InMemoryGameStore struct {
	gameTTL    time.Duration // How long a game is kept in memory.
	gameStates map[string]*gameStoreEntry
	// Sequence of game IDs. Used to list the most recent games.
	gameStatesSeq []string
	mut           sync.Mutex
	gamePubsubMap map[string]string
	subscribers   map[string][]chan<- *pb.GameStorePubsubEvent
}

type gameStoreEntry struct {
	gameState *pb.GameState
	created   time.Time
}

func NewInMemoryGameStore(gameTTL time.Duration) *InMemoryGameStore {
	return &InMemoryGameStore{
		gameTTL:       gameTTL,
		gameStates:    make(map[string]*gameStoreEntry),
		gamePubsubMap: make(map[string]string),
		subscribers:   make(map[string][]chan<- *pb.GameStorePubsubEvent),
	}
}

// Deletes all games from the store that are older than s.gameTTL.
// Must only be called while holding the s.mut lock.
func (s *InMemoryGameStore) deleteOldGames() {
	j := 0
	for i := 0; i < len(s.gameStatesSeq); i++ {
		id := s.gameStatesSeq[i]
		gs := s.gameStates[id]
		if time.Since(gs.created) >= s.gameTTL {
			delete(s.gameStates, id)
		} else {
			s.gameStatesSeq[j] = s.gameStatesSeq[i]
			j++
		}
	}
	s.gameStatesSeq = s.gameStatesSeq[:j]
}

func (s *InMemoryGameStore) StoreNewGame(ctx context.Context, state *pb.GameState) error {
	if state.GameInfo == nil {
		return fmt.Errorf("game state must have GameInfo")
	}
	s.mut.Lock()
	defer s.mut.Unlock()
	s.deleteOldGames()
	state.Modified = tpb.Now()
	gameId := state.GameInfo.Id
	if gameId == "" {
		for {
			gameId = GenerateGameID()
			if _, ok := s.gameStates[gameId]; !ok {
				break
			}
		}
	}
	state.GameInfo.Id = gameId
	pubsubId := state.GameInfo.PubsubChannelId
	if pubsubId == "" {
		pubsubId = GeneratePubsubID()
	}
	state.GameInfo.PubsubChannelId = pubsubId
	// Create a copy, like the remote store would, to avoid nasty concurrent access problems.
	s.gameStates[gameId] = &gameStoreEntry{
		gameState: proto.Clone(state).(*pb.GameState),
		created:   time.Now(),
	}
	s.gameStatesSeq = append(s.gameStatesSeq, gameId)
	return nil
}

func (s *InMemoryGameStore) LookupGame(ctx context.Context, gameId string) (*pb.GameState, error) {
	s.mut.Lock()
	defer s.mut.Unlock()
	if e, ok := s.gameStates[gameId]; ok {
		// Create a copy, like the remote store would, to avoid nasty concurrent access problems.
		state := proto.Clone(e.gameState).(*pb.GameState)
		return state, nil
	}
	return nil, ErrGameNotExist
}

func (s *InMemoryGameStore) UpdateGame(ctx context.Context, state *pb.GameState) error {
	s.mut.Lock()
	defer s.mut.Unlock()
	state.Modified = tpb.Now()
	gs, ok := s.gameStates[state.GetGameInfo().GetId()]
	if !ok {
		return fmt.Errorf("game %v does not exist in the store", state.GetGameInfo().GetId())
	}
	// Create a copy, like the remote store would, to avoid nasty concurrent access problems.
	gs.gameState = proto.Clone(state).(*pb.GameState)
	return nil
}

func (s *InMemoryGameStore) listGames(limit int, accept func(*pb.GameState) bool) ([]*GameInfo, error) {
	s.mut.Lock()
	defer s.mut.Unlock()
	s.deleteOldGames()
	l := len(s.gameStatesSeq)
	if limit > l {
		limit = l
	}
	var infos []*GameInfo
	for i := l - 1; i >= 0; i-- {
		id := s.gameStatesSeq[i]
		e := s.gameStates[id]
		if accept(e.gameState) {
			gi := e.gameState.GetGameInfo()
			infos = append(infos, &GameInfo{
				Id:       gi.Id,
				Host:     gi.Host,
				Started:  gi.Started.AsTime(),
				GameType: api.GameType(gi.Type),
			})
		}
		if len(infos) >= limit {
			break
		}
	}
	return infos, nil
}

func (s *InMemoryGameStore) ListOpenGames(ctx context.Context, limit int) ([]*GameInfo, error) {
	return s.listGames(limit, func(gs *pb.GameState) bool {
		return !gs.AllPlayersJoined
	})
}

func (s *InMemoryGameStore) ListActiveGames(ctx context.Context, limit int) ([]*GameInfo, error) {
	return s.listGames(limit, func(gs *pb.GameState) bool {
		return gs.AllPlayersJoined
	})
}

func (s *InMemoryGameStore) Publish(ctx context.Context, pubsubId string, event *pb.GameStorePubsubEvent) error {
	s.mut.Lock()
	defer s.mut.Unlock()
	for _, sub := range s.subscribers[pubsubId] {
		eventCopy := proto.Clone(event).(*pb.GameStorePubsubEvent)
		sub <- eventCopy
	}
	return nil
}

func (s *InMemoryGameStore) Subscribe(ctx context.Context, pubsubId string) <-chan *pb.GameStorePubsubEvent {
	s.mut.Lock()
	defer s.mut.Unlock()

	ch := make(chan *pb.GameStorePubsubEvent)
	sub := make(chan *pb.GameStorePubsubEvent)
	s.subscribers[pubsubId] = append(s.subscribers[pubsubId], sub)

	go func() {
		defer close(ch)
		defer func() {
			// Remove sub from subscriber list.
			s.mut.Lock()
			defer s.mut.Unlock()

			subs := s.subscribers[pubsubId]
			for i, s1 := range subs {
				if s1 == sub {
					l := len(subs)
					subs[i] = subs[l-1]
					subs = subs[:l-1]
					if len(subs) > 0 {
						s.subscribers[pubsubId] = subs
					} else {
						delete(s.subscribers, pubsubId)
					}
					return
				}
			}
		}()
		// Receive events on sub and forward them to ch.
		for {
			select {
			case event := <-sub:
				ch <- event
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch
}
