package hexz

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestDistribRangeLength(t *testing.T) {
	tests := []struct {
		name              string
		low, high, factor float64
		wantLen           int
	}{
		{"1-100/1.1", 1, 100, 1.1, 50},
		{"latency_1ms_1h", 0.001, 60 * 60, 1.1, 160},
		{"iterations", 1000, 1e9, 1.1, 146},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := DistribRange(test.low, test.high, test.factor)
			if len(got) != test.wantLen {
				t.Errorf("want: %v, got: %d (%v)", test.wantLen, len(got), got)
			}
		})
	}
}

func TestDistribution(t *testing.T) {
	d, err := NewDistribution("test", []float64{1, 2, 3})
	if err != nil {
		t.Fatalf("Failed to create distribution: %s", err.Error())
	}
	d.Add(-0.5)
	want := []int64{1, 0, 0, 0}
	if !cmp.Equal(d.counts, want) {
		t.Errorf("want %v, got %v", want, d.counts)
	}
	d.Add(1)
	want = []int64{1, 1, 0, 0}
	if !cmp.Equal(d.counts, want) {
		t.Errorf("want %v, got %v", want, d.counts)
	}
	d.Add(1.5)
	want = []int64{1, 2, 0, 0}
	if !cmp.Equal(d.counts, want) {
		t.Errorf("want %v, got %v", want, d.counts)
	}
	d.Add(3)
	want = []int64{1, 2, 0, 1}
	if !cmp.Equal(d.counts, want) {
		t.Errorf("want %v, got %v", want, d.counts)
	}
	d.Add(4)
	want = []int64{1, 2, 0, 2}
	if !cmp.Equal(d.counts, want) {
		t.Errorf("want %v, got %v", want, d.counts)
	}
	if d.totalCount != 5 {
		t.Errorf("want %v elements, got %v", 5, d.totalCount)
	}
	if d.min != -0.5 {
		t.Errorf("want min: %v, got %v", -0.5, d.min)
	}
	if d.max != 4 {
		t.Errorf("want max: %v, got %v", -0.5, d.max)
	}
	if d.sum != 9 {
		t.Errorf("want sum: %v, got %v", 9, d.sum)
	}
}
