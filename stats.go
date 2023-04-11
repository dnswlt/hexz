package hexz

import "sync"

type Counter struct {
	// Counter's name. Should start with and use "/" to create hierarchies,
	// e.g. "/storage/files/read".
	name string
	// Counter's value. Only access this via Value() to avoid data races.
	value int64
	mut   sync.Mutex
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
