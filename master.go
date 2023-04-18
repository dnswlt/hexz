package hexz

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"time"
)

type pInfo struct {
	playerNum int
	Player
}

type GameMaster struct {
	s          *Server
	game       *GameHandle
	randSrc    rand.Source
	gameEngine GameEngine
	// Player and spectator channels.
	eventListeners map[PlayerId]chan ServerEvent
	// Players registered for this game.
	players map[PlayerId]pInfo
	// Channels to coordinate delays of ending the game after a player left.
	// They can rejoin for a while, e.g. to gracefully handle page reloads
	// and accidentally closed tabs.
	removePlayer       chan PlayerId
	removePlayerCancel map[PlayerId]chan tok
	// Channel to request a move by the CPU player. Nil for 2P games.
	cpuCh chan tok
}

const (
	playerIdCPU = "CPU"
)

func (m *GameMaster) playerNames() []string {
	r := make([]string, len(m.players))
	for _, p := range m.players {
		r[p.playerNum-1] = p.Name
	}
	return r
}

func (m *GameMaster) broadcastPing(debugMessage string) {
	now := time.Now().Format(time.RFC3339)
	for _, ch := range m.eventListeners {
		ch <- ServerEvent{Timestamp: now, DebugMessage: debugMessage}
	}
}

func (m *GameMaster) broadcast(e *ServerEvent) {
	e.Timestamp = time.Now().Format(time.RFC3339)
	if m.gameEngine.Board().State != Initial {
		e.PlayerNames = m.playerNames()
	}
	// Send event to all listeners. Avoid recomputing board for spectators.
	var spectatorBoard *BoardView
	for pId, ch := range m.eventListeners {
		pNum := m.players[pId].playerNum
		if pNum > 0 {
			e.Board = m.gameEngine.Board().ViewFor(pNum)
			e.Role = pNum
		} else {
			if spectatorBoard == nil {
				spectatorBoard = m.gameEngine.Board().ViewFor(0)
			}
			e.Board = spectatorBoard
		}
		ch <- *e
	}
}

func (m *GameMaster) processControlEventRegister(e ControlEventRegister) {
	var playerNum int
	added := false
	if _, ok := m.players[e.player.Id]; ok {
		// Player reconnected. Cancel its removal.
		if cancel, ok := m.removePlayerCancel[e.player.Id]; ok {
			close(cancel)
			delete(m.removePlayerCancel, e.player.Id)
		}
	} else if len(m.players) < m.gameEngine.NumPlayers() {
		added = true
		playerNum = len(m.players) + 1
		m.players[e.player.Id] = pInfo{playerNum, e.player}
		if m.game.singlePlayer {
			m.cpuCh = make(chan tok)
			go m.cpuPlayer(playerIdCPU)
			m.players[playerIdCPU] =
				pInfo{playerNum: 2, Player: Player{Id: playerIdCPU, Name: "Computer"}}
		}
	}
	ch := make(chan ServerEvent)
	m.eventListeners[e.player.Id] = ch
	e.replyChan <- ch
	announcements := []string{}
	if added {
		announcements = append(announcements, fmt.Sprintf("Welcome %s!", e.player.Name))
	}
	if added && len(m.players) == m.gameEngine.NumPlayers() {
		announcements = append(announcements, "The game begins!")
	}
	m.broadcast(&ServerEvent{Announcements: announcements})
}

func (m *GameMaster) processControlEventUnregister(e ControlEventUnregister) {
	if ch, ok := m.eventListeners[e.playerId]; ok {
		// Close the eventListener channel. The receiver will already be gone anyway,
		// but for testing it's helpful to have that signal.
		close(ch)
		delete(m.eventListeners, e.playerId)
	}
	if _, ok := m.removePlayerCancel[e.playerId]; ok {
		// A repeated unregister should not happen. If it does, we ignore
		// it and just wait for the existing scheduled removal to trigger.
		return
	}
	if _, ok := m.players[e.playerId]; ok {
		// Remove player after timeout. Don't remove them immediately as they might
		// just be reloading their page and rejoin soon.
		cancel := make(chan tok)
		m.removePlayerCancel[e.playerId] = cancel
		go func(playerId PlayerId) {
			t := time.After(m.s.config.PlayerRemoveDelay)
			select {
			case <-t:
				m.removePlayer <- playerId
			case <-cancel:
			}
		}(e.playerId)
	}
}

func (m *GameMaster) processControlEventMove(e ControlEventMove) {
	p, ok := m.players[e.playerId]
	if !ok || m.gameEngine.Board().State != Running {
		// Ignore invalid move request
		return
	}
	if m.s.config.DebugMode {
		debugReq, _ := json.Marshal(e.MoveRequest)
		log.Printf("%s: move request: P%d %s", m.game.id, p.playerNum, debugReq)
	}
	if m.gameEngine.MakeMove(GameEngineMove{playerNum: p.playerNum, move: e.Move, row: e.Row, col: e.Col, cellType: e.Type}) {
		evt := &ServerEvent{Announcements: []string{}}
		if m.gameEngine.IsDone() {
			m.s.IncCounter(fmt.Sprintf("/games/%s/finished", m.game.gameType))
			if winner := m.gameEngine.Winner(); winner > 0 {
				evt.Winner = winner
				winnerName := m.playerNames()[winner-1]
				evt.Announcements = append(evt.Announcements,
					fmt.Sprintf("&#127942; &#127942; &#127942; %s won &#127942; &#127942; &#127942;",
						winnerName))
			}
		}
		if e.confidence > 0 {
			evt.Announcements = append(evt.Announcements, fmt.Sprintf("CPU confidence: %.3f", e.confidence))
		}
		if m.game.singlePlayer && m.gameEngine.Board().Turn != 1 && !m.gameEngine.IsDone() {
			// Ask CPU player to make a move.
			m.cpuCh <- tok{}
		}
		m.broadcast(evt)
	}
}

func (m *GameMaster) processControlEventReset(e ControlEventReset) {
	p, ok := m.players[e.playerId]
	if !ok {
		return // Only players are allowed to reset
	}
	m.gameEngine.Reset()
	announcements := []string{
		fmt.Sprintf("Player %s restarted the game.", p.Name),
	}
	m.broadcast(&ServerEvent{Announcements: announcements})
}

func (m *GameMaster) Run() {
	m.s.IncCounter(fmt.Sprintf("/games/%s/started", m.game.gameType))
	log.Printf("Started new %q game: %s", m.game.gameType, m.game.id)
	defer func() {
		close(m.game.done)
		m.s.deleteGame(m.game.id)
		// Signal that client SSE connections should be terminated.
		for _, ch := range m.eventListeners {
			close(ch)
		}
		if m.cpuCh != nil {
			close(m.cpuCh)
		}
	}()

	for {
		tick := time.After(5 * time.Second)
		select {
		case ce := <-m.game.controlEvent:
			switch e := ce.(type) {
			case ControlEventRegister:
				m.processControlEventRegister(e)
			case ControlEventUnregister:
				m.processControlEventUnregister(e)
			case ControlEventMove:
				m.processControlEventMove(e)
			case ControlEventReset:
				m.processControlEventReset(e)
			case ControlEventKill:
				m.broadcast(&ServerEvent{
					// Send sad emoji.
					Announcements: []string{"The game was terminated."},
				})
				return
			}
		case <-tick:
			m.broadcastPing("ping")
		case playerId := <-m.removePlayer:
			log.Printf("Player %s left game %s: game over", playerId, m.game.id)
			playerName := "?"
			if p, ok := m.players[playerId]; ok {
				playerName = p.Name
			}
			m.broadcast(&ServerEvent{
				// Send sad emoji.
				Announcements: []string{fmt.Sprintf("Player %s left the game &#128546;. Game over.", playerName)},
			})
			return
		}
	}
}

// Controller function for a running game. To be executed by a dedicated goroutine.
func NewGameMaster(s *Server, game *GameHandle) *GameMaster {
	randSrc := rand.NewSource(time.Now().UnixNano())
	m := &GameMaster{
		s:                  s,
		game:               game,
		randSrc:            randSrc,
		gameEngine:         NewGameEngine(game.gameType, randSrc),
		eventListeners:     make(map[PlayerId]chan ServerEvent),
		players:            make(map[PlayerId]pInfo),
		removePlayer:       make(chan PlayerId),
		removePlayerCancel: make(map[PlayerId]chan tok),
	}
	return m
}

func (m *GameMaster) cpuPlayer(cpuPlayerId PlayerId) {
	gameType := m.gameEngine.GameType()
	mcts := NewMCTS()
	// Minimum time to spend thinking about a move, even if we're dead certain about the result.
	minTime := time.Duration(100) * time.Millisecond
	thinkTime := m.s.config.CompThinkTime
	ge, ok := m.gameEngine.(SinglePlayerGameEngine)
	if !ok {
		log.Printf("Tried to create a CPU player for a non-CPU game type: %s", m.gameEngine.GameType())
		return
	}
	t := m.s.config.CompThinkTime
	for range m.cpuCh {
		mv, stats := mcts.SuggestMove(ge, t)
		if minQ := stats.MinQ(); minQ >= 0.98 || minQ <= 0.02 {
			// Speed up if we think we (almost) won or lost.
			t = t / 2
			if t < minTime {
				t = minTime
			}
		} else {
			t = thinkTime // use full time allowed.
		}
		// Send move request
		m.game.controlEvent <- ControlEventMove{
			playerId:   cpuPlayerId,
			confidence: stats.MaxQ(),
			MoveRequest: MoveRequest{
				Move: mv.move,
				Row:  mv.row,
				Col:  mv.col,
				Type: mv.cellType,
			},
		}
		// Update counters
		m.s.IncCounter(fmt.Sprintf("/games/%s/mcts/suggested_moves", gameType))
		if stats.FullyExplored {
			m.s.IncCounter(fmt.Sprintf("/games/%s/mcts/fully_explored", gameType))
		}
		m.s.AddDistribValue(fmt.Sprintf("/games/%s/mcts/elapsed", gameType), stats.Elapsed.Seconds())
		m.s.AddDistribValue(fmt.Sprintf("/games/%s/mcts/iterations", gameType), float64(stats.Iterations))
		m.s.AddDistribValue(fmt.Sprintf("/games/%s/mcts/tree_size", gameType), float64(stats.TreeSize))
		m.s.AddDistribValue(fmt.Sprintf("/games/%s/mcts/iterations_per_sec", gameType), float64(stats.Iterations)/stats.Elapsed.Seconds())
	}
}
