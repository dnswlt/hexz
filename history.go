package hexz

import (
	"encoding/json"
	"os"
	"path"
)

type GameHistoryEntry struct {
	Board      *BoardView
	MoveScores *MoveScores
}

func ReadGameHistory(historyDir string, gameId string) ([]*GameHistoryEntry, error) {
	data, err := os.ReadFile(path.Join(historyDir, gameId+".json"))
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
	w, err := os.Create(path.Join(historyDir, gameId+".json"))
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
