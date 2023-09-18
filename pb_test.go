package hexz

import (
	"testing"

	pb "github.com/dnswlt/hexz/hexzpb"
)

func TestCellTypeEnumsAligned(t *testing.T) {
	wantLen := int(cellTypeLen)
	if gotLen := len(pb.Field_CellType_name); wantLen != gotLen {
		t.Errorf("CellType enums are not aligned: want %d, got %d", wantLen, gotLen)
	}
	expectedPairs := []struct {
		p pb.Field_CellType
		e CellType
	}{
		{pb.Field_NORMAL, cellNormal},
		{pb.Field_DEAD, cellDead},
		{pb.Field_GRASS, cellGrass},
		{pb.Field_ROCK, cellRock},
		{pb.Field_FIRE, cellFire},
		{pb.Field_FLAG, cellFlag},
		{pb.Field_PEST, cellPest},
		{pb.Field_DEATH, cellDeath},
	}
	if len(expectedPairs) != int(cellTypeLen) {
		t.Errorf("Please update expectedPairs with added enums")
	}
	for _, p := range expectedPairs {
		if int(p.p) != int(p.e) {
			t.Errorf("CellType enum mismatch for %s", p.p.String())
		}
	}
}
