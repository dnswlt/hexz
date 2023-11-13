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
	if err := ExportSVG(path.Join(dir, "test_svg.html"), []*Board{flagz.B}, []string{"Test SVG"}); err != nil {
		t.Fatal(err)
	}
}

func TestConvertHexColor(t *testing.T) {
	tests := []struct {
		name    string
		col1    string
		col2    string
		scale   float64
		want    string
		wantErr bool
	}{
		{"middle_gray", "#000000", "#ffffff", 0.5, "#808080", false},
		{"varied", "#123456", "#abcdef", 0.3, "#406284", false},
		{"equal", "#abcdef", "#abcdef", 0.5, "#abcdef", false},
		{"equal0", "#abcdef", "#abcdef", 0.0, "#abcdef", false},
		{"equal1", "#abcdef", "#abcdef", 1.0, "#abcdef", false},
		{"invalid_scale", "#abcdef", "#abcdff", 3.0, "", true},
		{"negative", "#808080", "#efefef", -0.1, "#757575", false},
		{"wrong_input", "rgb(123, 123, 123)", "abcdef", 0, "", true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ScaleRGB(test.col1, test.col2, test.scale)
			if err == nil && test.wantErr {
				t.Error("Wanted error, but got none")
			}
			if err != nil && !test.wantErr {
				t.Error(err)
			}
			if got != test.want && !test.wantErr {
				t.Errorf("Got color %s, want: %s", got, test.want)
			}
		})
	}
}
