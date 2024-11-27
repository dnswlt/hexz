package hexz

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"
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

func TestMarshalTime(t *testing.T) {
	// What does a time.Time get JSON-marshalled to?
	type tm struct {
		T time.Time
	}
	loc, err := time.LoadLocation("Europe/Zurich")
	if err != nil {
		t.Fatal("cannot load location: ", err)
	}
	data, _ := json.Marshal(tm{time.Date(2023, 12, 31, 23, 59, 0, 123_000_000, loc)})
	want := `{"T":"2023-12-31T23:59:00.123+01:00"}`
	if string(data) != want {
		t.Errorf("Want: %s, got: %s", want, string(data))
	}
}

func TestFormatTimeRFC(t *testing.T) {
	loc, err := time.LoadLocation("Europe/Zurich")
	if err != nil {
		t.Fatal("cannot load location: ", err)
	}
	got := time.Date(2023, 12, 31, 23, 59, 0, 123_000_000, loc).Format(time.RFC3339)
	want := `2023-12-31T23:59:00+01:00`
	if got != want {
		t.Errorf("Want: %s, got: %s", want, got)
	}
}
