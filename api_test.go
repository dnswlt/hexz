package hexz

import (
	"fmt"
	"testing"
)

func TestValidCellTypes(t *testing.T) {
	tests := []struct {
		value CellType
		want  bool
	}{
		{-1, false},
		{cellNormal, true},
		{cellDead, true},
		{cellGrass, true},
		{cellRock, true},
		{cellFire, true},
		{cellFlag, true},
		{cellPest, true},
		{cellDeath, true},
		{cellTypeLen, false},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if got := test.value.valid(); got != test.want {
				t.Errorf("unexpected result %t for value %v", got, test.value)
			}
		})
	}
}

func TestCellTypesValuesAreConstant(t *testing.T) {
	// Once we persist games, we must no longer change the
	// numeric value of a CellType, as otherwise loading an
	// old game will break. This test reminds us of that.
	tests := []struct {
		value CellType
		want  int
	}{
		{cellNormal, 0},
		{cellDead, 1},
		{cellGrass, 2},
		{cellRock, 3},
		{cellFire, 4},
		{cellFlag, 5},
		{cellPest, 6},
		{cellDeath, 7},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if got := int(test.value); got != test.want {
				t.Errorf("Numeric value of CellType has changed: want: %d, got: %d", test.want, got)
			}
		})
	}
}
