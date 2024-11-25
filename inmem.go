package hexz

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"sync"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
	"github.com/dnswlt/hexz/hlog"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

// When run as a standalone app, we can store all logged in users
// in memory and sporadically dump the list to disk.
// When running as a container, we can't do that, so we need to
// store logins in some external storage.

const (
	maxLoggedInPlayers = 10000
)

var (
	errPlayerNotFound = errors.New("player not found")
)

type InMemoryPlayerStore struct {
	// Contains all logged in players, mapped by their (cookie) playerId.
	players  map[PlayerId]*Player
	mut      sync.Mutex
	loginTTL time.Duration // How long a login is valid.
	// Configuration for persistent storage.
	dbPath      string // Path to the file where the player DB is stored. If empty, no persistent storage is used.
	lastCleanup time.Time
}

// Creates a new in-memory player store and loads the player DB from the given file.
// If dbPath is empty, no persistent storage is used.
func NewInMemoryPlayerStore(loginTTL time.Duration, dbPath string) (*InMemoryPlayerStore, error) {
	s := &InMemoryPlayerStore{
		players:  make(map[PlayerId]*Player),
		dbPath:   dbPath,
		loginTTL: loginTTL,
	}
	if dbPath != "" {
		if err := s.loadFromFile(); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (s *InMemoryPlayerStore) Lookup(ctx context.Context, playerId PlayerId) (Player, error) {
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
		if s.dbPath != "" {
			go func() {
				if err := s.saveToFile(); err != nil {
					hlog.Errorf("Failed to save player DB: %v", err)
				}
			}()
		}
	}
	p, ok := s.players[playerId]
	if !ok {
		return Player{}, errPlayerNotFound
	}
	p.LastActive = time.Now()
	return *p, nil
}

func (s *InMemoryPlayerStore) Login(ctx context.Context, playerId PlayerId, name string) error {
	s.mut.Lock()
	defer s.mut.Unlock()
	if len(s.players) >= maxLoggedInPlayers {
		return fmt.Errorf("too many logged in players")
	}
	s.players[playerId] = &Player{
		Id:         playerId,
		Name:       name,
		LastActive: time.Now(),
	}
	return nil
}

func (s *InMemoryPlayerStore) NumPlayers() int {
	s.mut.Lock()
	defer s.mut.Unlock()
	return len(s.players)
}

func (s *InMemoryPlayerStore) loadFromFile() error {
	r, err := os.Open(s.dbPath)
	if errors.Is(err, fs.ErrNotExist) {
		return nil // No file yet, that's fine.
	}
	if err != nil {
		return err
	}
	defer r.Close()
	dec := json.NewDecoder(r)
	var players []*Player
	if err := dec.Decode(&players); err != nil {
		return fmt.Errorf("corrupted user database: %w", err)
	}
	s.mut.Lock()
	defer s.mut.Unlock()
	for _, p := range players {
		if _, ok := s.players[p.Id]; !ok {
			// Only add players, don't overwrite anything already in memory.
			s.players[p.Id] = p
		}
	}
	return nil
}

func (s *InMemoryPlayerStore) saveToFile() error {
	w, err := os.Create(s.dbPath)
	if err != nil {
		return err
	}
	defer w.Close()
	enc := json.NewEncoder(w)
	players := func() []*Player {
		s.mut.Lock()
		defer s.mut.Unlock()
		ps := make([]*Player, 0, len(s.players))
		for _, p := range s.players {
			ps = append(ps, p)
		}
		return ps
	}()
	if err := enc.Encode(players); err != nil {
		return err
	}
	return nil
}

var (
	errGameNotExist = fmt.Errorf("game does not exist in store")
)

// An in-memory replacement for a RedisClient, to enable purely local, single-server
// gameplay. The implementation does not clean up old games and should only be used
// for "casual play" at home, not in a public server environment!
type InMemoryGameStore struct {
	gameStates map[string]*pb.GameState
	// Sequence of game IDs. Used to list the most recent games.
	gameStatesSeq []string
	mut           sync.Mutex
	subscribers   map[string][]chan<- *pb.GameStorePubsubEvent
}

func NewInMemoryGameStore() *InMemoryGameStore {
	return &InMemoryGameStore{
		gameStates:  make(map[string]*pb.GameState),
		subscribers: make(map[string][]chan<- *pb.GameStorePubsubEvent),
	}
}

func (s *InMemoryGameStore) StoreNewGame(ctx context.Context, state *pb.GameState) (bool, error) {
	s.mut.Lock()
	defer s.mut.Unlock()
	state.Modified = tpb.Now()
	gameId := state.GetGameInfo().GetId()
	if _, ok := s.gameStates[gameId]; ok {
		return false, nil
	}
	// Create a copy, like the remote store would, to avoid nasty concurrent access problems.
	stateCopy := proto.Clone(state).(*pb.GameState)
	s.gameStates[gameId] = stateCopy
	s.gameStatesSeq = append(s.gameStatesSeq, gameId)
	return true, nil
}

func (s *InMemoryGameStore) LookupGame(ctx context.Context, gameId string) (*pb.GameState, error) {
	s.mut.Lock()
	defer s.mut.Unlock()
	if state, ok := s.gameStates[gameId]; ok {
		// Create a copy, like the remote store would, to avoid nasty concurrent access problems.
		stateCopy := proto.Clone(state).(*pb.GameState)
		return stateCopy, nil
	}
	return nil, errGameNotExist
}

func (s *InMemoryGameStore) UpdateGame(ctx context.Context, state *pb.GameState) error {
	s.mut.Lock()
	defer s.mut.Unlock()
	state.Seqnum++
	state.Modified = tpb.Now()
	// Create a copy, like the remote store would, to avoid nasty concurrent access problems.
	stateCopy := proto.Clone(state).(*pb.GameState)
	s.gameStates[state.GetGameInfo().GetId()] = stateCopy
	return nil
}

func (s *InMemoryGameStore) ListRecentGames(ctx context.Context, limit int) ([]*pb.GameInfo, error) {
	s.mut.Lock()
	defer s.mut.Unlock()
	l := len(s.gameStatesSeq)
	if limit > l {
		limit = l
	}
	infos := []*pb.GameInfo{}
	for i := 0; i < limit; i++ {
		id := s.gameStatesSeq[l-i-1]
		if state, ok := s.gameStates[id]; ok {
			infos = append(infos, state.GetGameInfo())
		} else {
			return nil, fmt.Errorf("invariant broken: gameStatesSeq not in sync: %s", id)
		}
	}
	return infos, nil
}

func (s *InMemoryGameStore) Publish(ctx context.Context, gameId string, event *pb.GameStorePubsubEvent) error {
	s.mut.Lock()
	defer s.mut.Unlock()
	for _, sub := range s.subscribers[gameId] {
		eventCopy := proto.Clone(event).(*pb.GameStorePubsubEvent)
		sub <- eventCopy
	}
	return nil
}

func (s *InMemoryGameStore) Subscribe(ctx context.Context, gameId string) <-chan *pb.GameStorePubsubEvent {
	s.mut.Lock()
	defer s.mut.Unlock()

	ch := make(chan *pb.GameStorePubsubEvent)
	sub := make(chan *pb.GameStorePubsubEvent)
	s.subscribers[gameId] = append(s.subscribers[gameId], sub)

	go func() {
		defer close(ch)
		defer func() {
			// Remove sub from subscriber list.
			s.mut.Lock()
			defer s.mut.Unlock()

			subs := s.subscribers[gameId]
			for i, s1 := range subs {
				if s1 == sub {
					l := len(subs)
					subs[i] = subs[l-1]
					subs = subs[:l-1]
					if len(subs) > 0 {
						s.subscribers[gameId] = subs
					} else {
						delete(s.subscribers, gameId)
					}
					return
				}
			}
		}()
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
