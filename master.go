package hexz

import (
	"context"
	"fmt"
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
	removePlayerCancel map[PlayerId]context.CancelFunc
	cpuPlayer          CPUPlayer // Nil for 2P games.

	// If undo/redo is enabled, these fields contain the
	// steps that can be undone/redone.
	undo []GameEngine
	redo []GameEngine

	historyWriter *HistoryWriter
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

func (m *GameMaster) broadcastPing() {
	e := &ServerEvent{
		Timestamp:    time.Now(),
		DebugMessage: "ping",
	}
	for _, ch := range m.eventListeners {
		ch <- *e
	}
}

func (m *GameMaster) broadcast(e *ServerEvent) {
	e.Timestamp = time.Now()
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
			e.Role = 0
		}
		ch <- *e
	}
}

// Creates a ServerEvent in which fields are populated that are only relevant
// for initialization of the client.
func (m *GameMaster) makeInitialServerEvent(announcements []string) *ServerEvent {
	return &ServerEvent{
		Announcements: announcements,
		GameInfo: &ServerEventGameInfo{
			ValidCellTypes: m.gameEngine.ValidCellTypes(),
			GameType:       m.game.gameType,
		},
	}
}

func (m *GameMaster) processControlEventRegister(e ControlEventRegister) {
	var playerNum int
	added := false
	if _, ok := m.players[e.player.Id]; ok {
		// Player reconnected. Cancel its removal.
		if cancel, ok := m.removePlayerCancel[e.player.Id]; ok {
			cancel()
			delete(m.removePlayerCancel, e.player.Id)
		}
	} else if len(m.players) < m.gameEngine.NumPlayers() {
		added = true
		playerNum = len(m.players) + 1
		m.players[e.player.Id] = pInfo{playerNum, e.player}
		if m.game.singlePlayer {
			m.cpuPlayer = NewLocalCPUPlayer(playerIdCPU, m.s.config.CpuThinkTime)
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
		m.historyWriter.WriteHeader(&GameHistoryHeader{
			GameId:      m.game.id,
			GameType:    m.game.gameType,
			PlayerNames: m.playerNames(),
		})
	}
	m.broadcast(m.makeInitialServerEvent(announcements))
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
		ctx, cancel := context.WithCancel(m.game.ctx)
		m.removePlayerCancel[e.playerId] = cancel
		go func(ctx context.Context, playerId PlayerId) {
			t := time.NewTimer(m.s.config.PlayerRemoveDelay)
			select {
			case <-t.C:
				m.removePlayer <- playerId
			case <-ctx.Done():
				// Game over or removal was cancelled.
				t.Stop()
			}
		}(ctx, e.playerId)
	}
}

func (m *GameMaster) processControlEventMove(e ControlEventMove) {
	p, ok := m.players[e.playerId]
	if !ok || m.gameEngine.Board().State != Running {
		// Ignore invalid move request
		return
	}
	mr := e.moveRequest
	if m.gameEngine.MakeMove(GameEngineMove{PlayerNum: p.playerNum, Move: mr.Move, Row: mr.Row, Col: mr.Col, CellType: mr.Type}) {
		if m.s.config.EnableUndo {
			var histEntry GameEngine
			if he, ok := m.gameEngine.(SinglePlayerGameEngine); ok {
				histEntry = he.Clone(m.randSrc)
				m.undo = append(m.undo, histEntry)
				m.redo = nil
			}
		}
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
		if e.mctsStats != nil {
			evt.Announcements = append(evt.Announcements, fmt.Sprintf("CPU confidence: %.3f", e.mctsStats.MaxQ()))
		}
		if m.cpuPlayer != nil && m.gameEngine.Board().Turn != 1 && !m.gameEngine.IsDone() {
			// Ask CPU player to make a move.
			go func(ctx context.Context) {
				mv, err := m.cpuPlayer.SuggestMove(ctx, m.gameEngine.(SinglePlayerGameEngine))
				if err != nil {
					// TODO: the game will be stuck now. We should probably cancel it.
					errorLog.Print("Couldn't get CPU move: ", err)
					return
				}
				select {
				case m.game.controlEvent <- mv:
				case <-ctx.Done():
				}
			}(m.game.ctx)
		}
		var moveScores *MoveScores
		if e.mctsStats != nil {
			moveScores = e.mctsStats.MoveScores()
		}
		m.historyWriter.Write(&GameHistoryEntry{
			EntryType:  "move",
			Move:       e.moveRequest,
			Board:      m.gameEngine.Board().ViewFor(0),
			MoveScores: moveScores,
		})
		if m.gameEngine.IsDone() {
			// Make history available immediately. Do not close the writer yet,
			// because a subsequent Reset might lead to more entries being written.
			m.historyWriter.Flush()
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
	m.historyWriter.Write(&GameHistoryEntry{
		EntryType: "reset",
		Board:     m.gameEngine.Board().ViewFor(0),
	})
	announcements := []string{
		fmt.Sprintf("Player %s restarted the game.", p.Name),
	}
	m.broadcast(m.makeInitialServerEvent(announcements))
}

func (m *GameMaster) processControlEventUndo(e ControlEventUndo) {
	if len(m.undo) == 0 {
		return
	}
	_, ok := m.players[e.playerId]
	if !ok || m.gameEngine.Board().Move != e.Move {
		// Only undo if it was requested for the current move by one of the players.
		return
	}
	m.redo = append(m.redo, m.gameEngine)
	m.gameEngine = m.undo[len(m.undo)-1]
	m.undo = m.undo[:len(m.undo)-1]
	m.historyWriter.Write(&GameHistoryEntry{
		EntryType: "undo",
		Board:     m.gameEngine.Board().ViewFor(0),
	})
	m.broadcast(&ServerEvent{Announcements: []string{"Undo"}})
}

func (m *GameMaster) processControlEventRedo(e ControlEventRedo) {
	if len(m.redo) == 0 {
		return
	}
	_, ok := m.players[e.playerId]
	if !ok || m.gameEngine.Board().Move != e.Move {
		// Only undo if it was requested for the current move by one of the players.
		return
	}
	m.undo = append(m.undo, m.gameEngine)
	m.gameEngine = m.redo[len(m.redo)-1]
	m.redo = m.redo[:len(m.redo)-1]
	m.historyWriter.Write(&GameHistoryEntry{
		EntryType: "redo",
		Board:     m.gameEngine.Board().ViewFor(0),
	})
	m.broadcast(&ServerEvent{Announcements: []string{"Redo"}})
}

func (m *GameMaster) processControlEventValidMoves(e ControlEventValidMoves) {
	defer close(e.reply)
	if engine, ok := m.gameEngine.(SinglePlayerGameEngine); ok {
		validMoves := engine.ValidMoves()
		moves := make([]*MoveRequest, len(validMoves))
		for i, m := range validMoves {
			moves[i] = &MoveRequest{
				Move: m.Move,
				Row:  m.Row,
				Col:  m.Col,
				Type: m.CellType,
			}
		}
		e.reply <- moves
	}
}

func (m *GameMaster) Run(cancel context.CancelFunc) {
	m.s.IncCounter(fmt.Sprintf("/games/%s/started", m.game.gameType))
	infoLog.Printf("Started new %q game: %s", m.game.gameType, m.game.id)
	defer func() {
		cancel()
		for _, ch := range m.eventListeners {
			// SSE clients should react to context cancellation, but
			// it doesn't hurt to close these channels as well.
			close(ch)
		}
		// historyWriter is closed when the game is done, but we
		// also want to retain history when players leave mid-game.
		m.historyWriter.Close()
	}()
	var lastEventReceived time.Time
	for {
		tick := time.After(5 * time.Second)
		select {
		case ce := <-m.game.controlEvent:
			lastEventReceived = time.Now()
			switch e := ce.(type) {
			case ControlEventRegister:
				m.processControlEventRegister(e)
			case ControlEventUnregister:
				m.processControlEventUnregister(e)
			case ControlEventMove:
				m.processControlEventMove(e)
			case ControlEventReset:
				m.processControlEventReset(e)
			case ControlEventUndo:
				m.processControlEventUndo(e)
			case ControlEventRedo:
				m.processControlEventRedo(e)
			case ControlEventValidMoves:
				m.processControlEventValidMoves(e)
			}
		case <-tick:
			if time.Since(lastEventReceived) > m.s.config.InactivityTimeout {
				m.broadcast(&ServerEvent{
					Announcements: []string{"Game was terminated after 1 hour of inactivity."},
				})
				return
			}
			m.broadcastPing()
		case playerId := <-m.removePlayer:
			infoLog.Printf("Player %s left game %s: game over", playerId, m.game.id)
			playerName := "?"
			if p, ok := m.players[playerId]; ok {
				playerName = p.Name
			}
			m.broadcast(&ServerEvent{
				// Send sad emoji.
				Announcements: []string{fmt.Sprintf("Player %s left the game &#128546;. Game over.", playerName)},
			})
			return
		case <-m.game.ctx.Done():
			return // game was cancelled externally.
		}

	}
}

// Controller function for a running game. To be executed by a dedicated goroutine.
func NewGameMaster(s *Server, game *GameHandle) *GameMaster {
	randSrc := rand.NewSource(time.Now().UnixNano())
	var historyWriter *HistoryWriter
	if s.config.GameHistoryRoot != "" {
		var err error
		historyWriter, err = NewHistoryWriter(s.config.GameHistoryRoot, game.id)
		if err != nil {
			errorLog.Printf("Cannot create history writer for game %s: %s", game.id, err)
		}
	}
	ge := NewGameEngine(game.gameType, randSrc)
	m := &GameMaster{
		s:              s,
		game:           game,
		randSrc:        randSrc,
		gameEngine:     ge,
		eventListeners: make(map[PlayerId]chan ServerEvent),
		players:        make(map[PlayerId]pInfo),
		// Use a buffered channel to avoid blocking the cancellation goroutines in case the game ends before they call back.
		removePlayer:       make(chan PlayerId, ge.NumPlayers()),
		removePlayerCancel: make(map[PlayerId]context.CancelFunc),
		historyWriter:      historyWriter,
	}
	return m
}
