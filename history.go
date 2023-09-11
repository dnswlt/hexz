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
)

type GameHistoryHeader struct {
	GameId      string
	PlayerNames []string
}

type GameHistoryEntry struct {
	Board      *BoardView
	MoveScores *MoveScores
}

// GameHistoryRecord is the struct that is persisted on disk
// as a sequence of gobs.
type GameHistoryRecord struct {
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

func (w *HistoryWriter) WriteHeader(header *GameHistoryHeader) error {
	if w.numRecords != 0 {
		return fmt.Errorf("header must be the first record written")
	}
	w.numRecords++
	return w.enc.Encode(GameHistoryRecord{Header: header})
}

func (w *HistoryWriter) Write(entry *GameHistoryEntry) error {
	w.numRecords++
	return w.enc.Encode(GameHistoryRecord{Entry: entry})
}

func (w *HistoryWriter) Close() error {
	if w.closed {
		return nil
	}
	w.closed = true
	if err := w.gz.Close(); err != nil {
		return err
	}
	return w.w.Close()
}

func ReadGameHistory(historyDir string, gameId string) (*GameHistoryHeader, []*GameHistoryEntry, error) {
	f, err := os.Open(path.Join(historyDir, gameIdPath(gameId)))
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, err
	}
	dec := gob.NewDecoder(r)
	var header *GameHistoryHeader
	entries := []*GameHistoryEntry{}
	lastMove := -1
	for {
		var record GameHistoryRecord
		err := dec.Decode(&record)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if record.Header != nil {
			header = record.Header
		} else if record.Entry != nil {
			// Handle undo/redo
			lastMove = record.Entry.Board.Move
			if lastMove < len(entries) {
				// Undo
				entries[lastMove] = record.Entry
			} else if lastMove == len(entries) {
				// Regular move
				entries = append(entries, record.Entry)
			} else {
				// We don't expect to see move N if we never saw move N-1.
				return nil, nil, fmt.Errorf("invalid history: move jumped to %d, expected at most %d",
					lastMove, len(entries))
			}
		}
	}
	return header, entries[:lastMove+1], nil
}
