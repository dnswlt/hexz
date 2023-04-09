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
		{"123", true},
		{"ab", false},
		{"", false},
		{"_letterordigit", false},
		{"verylongusernamesarenotallowedalright", false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isValidPlayerName(test.name); got != test.want {
				t.Errorf("unexpected result %t for name %s", test.want, test.name)
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
