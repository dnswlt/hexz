package hexz

import (
	"compress/gzip"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
	"time"
)

type GameHistory struct {
	Header  *GameHistoryHeader
	Entries []*GameHistoryEntry
}

type GameHistoryHeader struct {
	GameId      string
	GameType    GameType
	PlayerNames []string
}

type GameHistoryEntry struct {
	Timestamp  time.Time // Will be added automatically by the .Write method if not specified.
	EntryType  string    // One of {"move", "undo", "redo", "reset"}.
	Move       *MoveRequest
	Board      *BoardView
	MoveScores *MoveScores
}

// gameHistoryRecord is the struct that is persisted on disk
// as a sequence of gobs.
type gameHistoryRecord struct {
	// Only one of the fields will ever be populated.
	Header *GameHistoryHeader
	Entry  *GameHistoryEntry
}

type HistoryWriter struct {
	w          io.WriteCloser
	gz         *gzip.Writer
	enc        *gob.Encoder
	numRecords int // Number of records written so far
	closed     bool
}

// Returns a relative file path for the given gameId.
// Games are stored in subdirectories named after the first two uppercase(d) letters.
func gameIdPath(gameId string) string {
	if gameId == "" {
		gameId = "_"
	}
	if len(gameId) < 2 {
		return fmt.Sprintf("%s/%s.ggz", strings.ToUpper(gameId), gameId)
	}
	dir := strings.ToUpper(gameId[:2])
	return fmt.Sprintf("%s/%s.ggz", dir, gameId)
}

func NewHistoryWriter(historyDir, gameId string) (*HistoryWriter, error) {
	p := path.Join(historyDir, gameIdPath(gameId))
	err := os.MkdirAll(path.Dir(p), 0755)
	if err != nil {
		return nil, err
	}
	f, err := os.Create(p)
	if err != nil {
		return nil, err
	}
	gz := gzip.NewWriter(f)
	enc := gob.NewEncoder(gz)
	return &HistoryWriter{
		w:   f,
		gz:  gz,
		enc: enc,
	}, nil
}

// Writes the given header to the writer's game history.
// This method must be called only once, and before any calls to Write.
// w may be a nil receiver, in which case this method does nothing.
func (w *HistoryWriter) WriteHeader(header *GameHistoryHeader) error {
	if w == nil {
		return nil
	}
	if w.numRecords != 0 {
		return fmt.Errorf("header must be the first record written")
	}
	w.numRecords++
	return w.enc.Encode(gameHistoryRecord{Header: header})
}

// Appends the given entry to the writer's game history.
// w may be a nil receiver, in which case this method does nothing.
func (w *HistoryWriter) Write(entry *GameHistoryEntry) error {
	if w == nil {
		return nil
	}
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}
	w.numRecords++
	return w.enc.Encode(gameHistoryRecord{Entry: entry})
}

func (w *HistoryWriter) Close() error {
	if w == nil {
		return nil
	}
	if w.closed {
		return nil
	}
	w.closed = true
	if err := w.gz.Close(); err != nil {
		return err
	}
	return w.w.Close()
}

func ReadGameHistory(historyDir string, gameId string) (*GameHistory, error) {
	f, err := os.Open(path.Join(historyDir, gameIdPath(gameId)))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	dec := gob.NewDecoder(r)
	hist := &GameHistory{}
	for {
		var record gameHistoryRecord
		err := dec.Decode(&record)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		if record.Header != nil {
			hist.Header = record.Header
		} else if record.Entry != nil {
			hist.Entries = append(hist.Entries, record.Entry)
		}
	}
	return hist, nil
}
