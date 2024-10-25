package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/dnswlt/hexz"
)

func main() {
	cfg := &hexz.MoveSuggesterServerConfig{}

	flag.StringVar(&cfg.Addr, "addr", "localhost:50051", "Address on which to listen")
	flag.DurationVar(&cfg.CpuThinkTime, "cpu-think-time", 5*time.Second,
		"Time the computer has to think about a move")
	flag.IntVar(&cfg.CpuMaxFlags, "cpu-max-flags", 5,
		"Maximum flag moves to consider in any turn. <= 0 means unlimited")
	flag.Parse()

	if len(flag.Args()) > 0 {
		fmt.Fprintf(os.Stderr, "unexpected extra arguments: %v\n", flag.Args())
		os.Exit(1)
	}

	log.Fatal(hexz.NewMoveSuggesterServer(cfg).Serve())
}
