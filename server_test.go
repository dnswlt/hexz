package hexz

import "testing"

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
