package hexz

import (
	"fmt"
	"testing"
)

func TestValidPlayerName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"abc", true},
		{"abc.def", true},
		{"abc_def-123", true},
		{"1digit", true},
		{"HANS", true},
		{"Mørän", true},
		{"Jérôme", true},
		{"Strüßenbähn", true},
		{"123", true},
		{"_letter-or.digit", true},
		{"ab", false},      // Too short
		{"jens$", false},   // Invalid character
		{"dw@best", false}, // Invalid character
		{"", false},
		{"verylongusernamesarenotallowedalright", false},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if got := isValidPlayerName(test.name); got != test.want {
				t.Errorf("unexpected result %t for name %s", got, test.name)
			}
		})
	}
}

func TestValidCellTypes(t *testing.T) {
	tests := []struct {
		value CellType
		want  bool
	}{
		{-1, false},
		{cellNormal, true},
		{cellDead, true},
		{cellRock, true},
		{cellFire, true},
		{cellFlag, true},
		{cellPest, true},
		{cellDeath, true},
		{cellDeath + 1, false},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if got := test.value.valid(); got != test.want {
				t.Errorf("unexpected result %t for value %v", got, test.value)
			}
		})
	}
}

func TestCopyLocalToHeap(t *testing.T) {
	type s struct {
		t string
	}

	var l s
	l.t = "hello"
	m := make(map[string]*s)
	m["a"] = &l
	l.t = "world"
	m["b"] = &l
	if m["a"].t != "hello" {
		t.Errorf("Want m[\"a\"] == \"hello\", got: %s", m["a"].t)
	}
	if m["b"].t != "world" {
		t.Errorf("Want m[\"b\"] == \"world\", got: %s", m["b"].t)
	}
}

func TestSha256HexDigest(t *testing.T) {
	got := sha256HexDigest("foo")
	want := "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
	if got != want {
		t.Errorf("Want: %q, got: %q", want, got)
	}
}
