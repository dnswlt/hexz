package hexz

import "sync"

type Counter struct {
	name  string
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
