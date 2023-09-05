package hexz

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"
)

type GameHistoryEntry struct {
	Board      *BoardView
	MoveScores *MoveScores
}

// Returns a relative file path for the given gameId.
// Games are stored in subdirectories named after the first two uppercase(d) letters.
func gameIdPath(gameId string) string {
	if gameId == "" {
		gameId = "_"
	}
	if len(gameId) < 2 {
		return fmt.Sprintf("%s/%s.json", strings.ToUpper(gameId), gameId)
	}
	dir := strings.ToUpper(gameId[:2])
	return fmt.Sprintf("%s/%s.json", dir, gameId)
}

func ReadGameHistory(historyDir string, gameId string) ([]*GameHistoryEntry, error) {
	data, err := os.ReadFile(path.Join(historyDir, gameIdPath(gameId)))
	if err != nil {
		return nil, err
	}
	entries := []*GameHistoryEntry{}
	err = json.Unmarshal(data, &entries)
	if err != nil {
		return nil, err
	}
	return entries, nil
}

func WriteGameHistory(historyDir string, gameId string, hist []*GameHistoryEntry) error {
	p := path.Join(historyDir, gameIdPath(gameId))
	err := os.MkdirAll(path.Dir(p), 0755)
	if err != nil {
		return err
	}
	w, err := os.Create(p)
	if err != nil {
		return err
	}
	defer w.Close()
	enc := json.NewEncoder(w)
	if err := enc.Encode(hist); err != nil {
		return err
	}
	return nil
}
