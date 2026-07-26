package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dnswlt/hexz/pkg/hexz"
	"github.com/dnswlt/hexz/pkg/hexzpb"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// StartingPosition is an immutable initial Flagz state used by the benchmark.
// ID is derived from State, so independently created corpora can still be
// joined by position.
type StartingPosition struct {
	ID    string
	State *hexzpb.GameEngineState
}

type positionJSON struct {
	ID    string          `json:"id"`
	State json.RawMessage `json:"state"`
}

func positionID(state *hexzpb.GameEngineState) (string, error) {
	data, err := proto.MarshalOptions{Deterministic: true}.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal position: %w", err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:8]), nil
}

func newStartingPosition(state *hexzpb.GameEngineState) (StartingPosition, error) {
	id, err := positionID(state)
	if err != nil {
		return StartingPosition{}, err
	}
	return StartingPosition{ID: id, State: state}, nil
}

func generateStartingPositions(n int) ([]StartingPosition, error) {
	if n <= 0 {
		return nil, fmt.Errorf("number of positions must be positive, got %d", n)
	}
	positions := make([]StartingPosition, 0, n)
	seen := make(map[string]bool, n)
	for len(positions) < n {
		p, err := newStartingPosition(hexz.NewGameEngineFlagz().Proto())
		if err != nil {
			return nil, err
		}
		if seen[p.ID] {
			continue
		}
		seen[p.ID] = true
		positions = append(positions, p)
	}
	return positions, nil
}

func writeStartingPositions(path string, positions []StartingPosition) error {
	if path == "" {
		return fmt.Errorf("position file path is empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create position directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("create position file: %w", err)
	}
	defer f.Close()

	protoOpts := protojson.MarshalOptions{UseProtoNames: true}
	w := bufio.NewWriter(f)
	for _, p := range positions {
		state, err := protoOpts.Marshal(p.State)
		if err != nil {
			return fmt.Errorf("marshal position %s: %w", p.ID, err)
		}
		line, err := json.Marshal(positionJSON{ID: p.ID, State: state})
		if err != nil {
			return fmt.Errorf("marshal position envelope %s: %w", p.ID, err)
		}
		if _, err := w.Write(append(line, '\n')); err != nil {
			return fmt.Errorf("write position %s: %w", p.ID, err)
		}
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush position file: %w", err)
	}
	return nil
}

func readStartingPositions(path string) ([]StartingPosition, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open position file: %w", err)
	}
	defer f.Close()

	var positions []StartingPosition
	seen := make(map[string]bool)
	scanner := bufio.NewScanner(f)
	for line := 1; scanner.Scan(); line++ {
		data := bytes.TrimSpace(scanner.Bytes())
		if len(data) == 0 || data[0] == '#' {
			continue
		}
		var envelope positionJSON
		if err := json.Unmarshal(data, &envelope); err != nil {
			return nil, fmt.Errorf("decode position file line %d: %w", line, err)
		}
		state := &hexzpb.GameEngineState{}
		if err := protojson.Unmarshal(envelope.State, state); err != nil {
			return nil, fmt.Errorf("decode state on line %d: %w", line, err)
		}
		position, err := newStartingPosition(state)
		if err != nil {
			return nil, fmt.Errorf("position on line %d: %w", line, err)
		}
		if envelope.ID != position.ID {
			return nil, fmt.Errorf(
				"position ID mismatch on line %d: got %q, computed %q",
				line, envelope.ID, position.ID,
			)
		}
		if seen[position.ID] {
			return nil, fmt.Errorf("duplicate position %q on line %d", position.ID, line)
		}
		seen[position.ID] = true
		positions = append(positions, position)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read position file: %w", err)
	}
	if len(positions) == 0 {
		return nil, fmt.Errorf("position file %q contains no positions", path)
	}
	return positions, nil
}
