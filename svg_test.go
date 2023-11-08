package hexz

import (
	"path"
	"testing"
)

func TestExportSVG(t *testing.T) {
	// Play a few moves and export the SVG.
	// Set dir to sth non-temp to view an output in a browser.
	dir := t.TempDir()
	flagz := NewGameEngineFlagz()
	for i := 0; i < 10; i++ {
		m, err := flagz.RandomMove()
		if err != nil {
			t.Fatal(err)
		}
		if !flagz.MakeMove(m) {
			t.Fatal("Cannot make move")
		}
	}
	if err := ExportSVG(path.Join(dir, "test_svg.html"), []*Board{flagz.B}, nil, []string{"Test SVG"}); err != nil {
		t.Fatal(err)
	}
}
