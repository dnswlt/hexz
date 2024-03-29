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
	zw         *gzip.Writer
	enc        *gob.Encoder
	numRecords int // Number of records written so far
	closed     bool
}

func gameIdPath(historyDir, gameId string) string {
	return path.Join(historyDir, gameIdFile(gameId))
}

// Returns a relative file path for the given gameId.
// Games are stored in subdirectories named after the first two uppercase(d) letters.
func gameIdFile(gameId string) string {
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
	p := gameIdPath(historyDir, gameId)
	err := os.MkdirAll(path.Dir(p), 0755)
	if err != nil {
		return nil, err
	}
	f, err := os.Create(p)
	if err != nil {
		return nil, err
	}
	zw := gzip.NewWriter(f)
	enc := gob.NewEncoder(zw)
	return &HistoryWriter{
		w:   f,
		zw:  zw,
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

func (w *HistoryWriter) Flush() error {
	if w == nil {
		return nil
	}
	if err := w.zw.Flush(); err != nil {
		return err
	}
	// if f, ok := w.w.(*os.File); ok {
	// 	// Sync is not strictly necessary, since .Write on an os.File is unbuffered.
	// 	if err := f.Sync(); err != nil {
	// 		return err
	// 	}
	// }
	return nil
}

func (w *HistoryWriter) Close() error {
	if w == nil {
		return nil
	}
	if w.closed {
		return nil
	}
	w.closed = true
	if err := w.zw.Close(); err != nil {
		return err
	}
	return w.w.Close()
}

func GameHistoryExists(historyDir string, gameId string) bool {
	fi, err := os.Stat(gameIdPath(historyDir, gameId))
	if err != nil {
		return false
	}
	return fi.Mode().IsRegular() && fi.Size() > 0
}

func ReadGameHistory(historyDir string, gameId string) (*GameHistory, error) {
	f, err := os.Open(gameIdPath(historyDir, gameId))
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
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			// Ignore ErrUnexpectedEOF as well, since that's what we'll get
			// when we try to read a history file that's flushed, but not closed yet.
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
