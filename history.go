package hexz

import (
	"bufio"
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
	w   io.WriteCloser
	buf *bufio.Writer
	enc *gob.Encoder
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
	buf := bufio.NewWriter(f)
	enc := gob.NewEncoder(buf)
	return &HistoryWriter{
		w:   f,
		buf: buf,
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
		return fmt.Sprintf("%s/%s.gob", strings.ToUpper(gameId), gameId)
	}
	dir := strings.ToUpper(gameId[:2])
	return fmt.Sprintf("%s/%s.gob", dir, gameId)
}

func ReadGameHistory(historyDir string, gameId string) ([]*GameHistoryEntry, error) {
	f, err := os.Open(path.Join(historyDir, gameIdPath(gameId)))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := gob.NewDecoder(bufio.NewReader(f))
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
	if err := w.buf.Flush(); err != nil {
		return err
	}
	return w.w.Close()
}
