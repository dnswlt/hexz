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

type GameHistoryEntry struct {
	Board      *BoardView
	MoveScores *MoveScores
}

type HistoryWriter struct {
	w      io.WriteCloser
	gz     *gzip.Writer
	enc    *gob.Encoder
	closed bool
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

func ReadGameHistory(historyDir string, gameId string) ([]*GameHistoryEntry, error) {
	f, err := os.Open(path.Join(historyDir, gameIdPath(gameId)))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r, _ := gzip.NewReader(f)
	// r, _ := gzip.NewReader(bufio.NewReader(f))
	dec := gob.NewDecoder(r)
	result := []*GameHistoryEntry{}
	for {
		var entry *GameHistoryEntry
		err := dec.Decode(&entry)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		result = append(result, entry)
	}
	return result, nil
}

func (w *HistoryWriter) Write(entry *GameHistoryEntry) error {
	return w.enc.Encode(entry)
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
