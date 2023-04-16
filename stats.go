package hexz

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type Counter struct {
	// Counter's name. Should start with and use "/" to create hierarchies,
	// e.g. "/storage/files/read".
	name string
	// Counter's value. Only access this via Value() to avoid data races.
	value int64
	mut   sync.Mutex
}

type Distribution struct {
	name        string
	counts      []int64
	upperBounds []float64
	totalCount  int64
	sum         float64
	min         float64
	max         float64
	mut         sync.Mutex
}

func NewCounter(name string) *Counter {
	return &Counter{name: name}
}

func (c *Counter) Increment() {
	c.mut.Lock()
	defer c.mut.Unlock()

	c.value++
}

func (c *Counter) Name() string {
	// A counter's name is immutable, so we don't synchronize here.
	return c.name
}

func (c *Counter) Value() int64 {
	c.mut.Lock()
	defer c.mut.Unlock()
	return c.value
}

func NewDistribution(name string, upperBounds []float64) (*Distribution, error) {
	if len(upperBounds) == 0 {
		return nil, fmt.Errorf("need at least one upper bound")
	}
	copiedBounds := make([]float64, len(upperBounds))
	copy(copiedBounds, upperBounds)
	for i := 0; i < len(copiedBounds)-1; i++ {
		if copiedBounds[i] >= copiedBounds[i+1] {
			return nil, fmt.Errorf("bounds must be strictly monotonically increasing")
		}
	}
	counts := make([]int64, len(copiedBounds)+1)
	return &Distribution{
		name:        name,
		upperBounds: copiedBounds,
		counts:      counts,
		min:         math.Inf(1),
		max:         math.Inf(-1),
	}, nil
}

// Returns a range of upper bounds, starting at low and ending at high (exactly),
// with factor increments, i.e. the result R has R[i] * factor = R[i+1],
// except for the last element, which is guaranteed to be high.
func DistribRange(low, high, factor float64) []float64 {
	if low >= high || factor <= 1.0 {
		return []float64{low}
	}
	r := []float64{}
	for v := low; v < high; v *= factor {
		r = append(r, v)
	}
	return append(r, high)
}

func (d *Distribution) Add(value float64) {
	d.mut.Lock()
	defer d.mut.Unlock()
	ix := sort.Search(len(d.upperBounds), func(i int) bool { return d.upperBounds[i] > value })
	d.counts[ix]++
	d.totalCount++
	d.sum += value
	if value < d.min {
		d.min = value
	}
	if value > d.max {
		d.max = value
	}
}

// For now, instead of providing synchronized access to all individual fields,
// just let clients copy the whole Distribution if they want to read it.
func (d *Distribution) Copy() *Distribution {
	d.mut.Lock()
	defer d.mut.Unlock()
	counts := make([]int64, len(d.counts))
	copy(counts, d.counts)
	upperBounds := make([]float64, len(d.upperBounds))
	copy(upperBounds, d.upperBounds)
	return &Distribution{
		name:        d.name,
		counts:      counts,
		upperBounds: upperBounds,
		totalCount:  d.totalCount,
		sum:         d.sum,
		min:         d.min,
		max:         d.max,
	}
}
