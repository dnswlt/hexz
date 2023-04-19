package hexz

// Testing some options for serialization of game states.

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"testing"
)

type foo interface {
	Honky() bool
}

type bar struct {
	X int
}

type quz struct {
	Y int
}

func (b bar) Honky() bool {
	return b.X > 0
}

func (q quz) Honky() bool {
	return false
}

type baz struct {
	F foo
}

func TestJsonMarshalInterfaceField(t *testing.T) {
	// We can marshal a struct that has a field of interface type.
	b := baz{F: bar{X: 3}}
	js, err := json.Marshal(b)
	if err != nil {
		t.Errorf("Marshal error: %s", err)
	}
	wantJson := `{"F":{"X":3}}`
	if s := string(js); s != wantJson {
		t.Errorf("Want %s, got %s", wantJson, s)
	}
}

func TestJsonUnmarshalInterfaceField(t *testing.T) {
	// We CANNOT unmarshal a struct that has a field of interface type.
	b := baz{F: bar{X: 3}}
	js, err := json.Marshal(b)
	if err != nil {
		t.Errorf("Marshal error: %s", err)
	}
	if err := json.Unmarshal(js, &b); err == nil {
		t.Error("Want error in Unmarshal, got none.")
	}
}

func TestGobEncodeInterfaceField(t *testing.T) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	gob.Register(bar{})
	gob.Register(quz{})
	b := baz{F: bar{X: 3}}
	err := enc.Encode(b)
	if err != nil {
		t.Errorf("Encode error: %s", err)
	}
	dec := gob.NewDecoder(&buf)
	var b1 baz
	if err := dec.Decode(&b1); err != nil {
		t.Errorf("Cannot decode: %s", err)
	}
	if a, ok := b1.F.(bar); !ok || a.X != 3 {
		t.Errorf("Not the value I expect: %+v", b1.F)
	}
	// One more time, now with quz:
	b = baz{F: quz{Y: 4}}
	err = enc.Encode(b)
	if err != nil {
		t.Errorf("Encode error: %s", err)
	}
	if err := dec.Decode(&b1); err != nil {
		t.Errorf("Cannot decode: %s", err)
	}
	if a, ok := b1.F.(quz); !ok || a.Y != 4 {
		t.Errorf("Not the value I expect: %+v", b1.F)
	}
}
