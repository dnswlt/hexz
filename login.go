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
)

// When run as a standalone app, we can store all logged in users
// in memory and sporadically dump the list to disk.
// When running as a container, we can't do that, so we need to
// store logins in some external storage.

// A random UUID used to identify players. Also used in cookies.
type PlayerId string

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
					errorLog.Print("Failed to save player DB: ", err)
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
