package hexz

import (
	"sort"
	"time"

	"github.com/dnswlt/hexz/internal/hlog"
)

type cpuThinkTimeOption struct {
	Label    string
	Millis   int64
	Selected bool
}

func cpuThinkTimeOptions(maxThinkTime time.Duration) []cpuThinkTimeOption {
	maxMillis := maxThinkTime.Milliseconds()
	options := []cpuThinkTimeOption{
		{Label: "100ms", Millis: 100},
		{Label: "1s", Millis: 1000},
		{Label: "3s", Millis: 3000},
		{Label: "5s", Millis: 5000},
		{Label: maxThinkTime.String(), Millis: maxMillis},
	}
	sort.Slice(options, func(i, j int) bool {
		return options[i].Millis < options[j].Millis
	})
	for i := 0; i < len(options); i++ {
		if options[i].Millis == maxMillis {
			options[i].Selected = true
			return options[:i+1]
		}
	}
	hlog.Fatalf("program error: fell off the loop")
	return nil
}
