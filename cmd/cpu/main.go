package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/dnswlt/hexz"
)

func main() {
	cfg := &hexz.CPUPlayerServerConfig{}

	flag.StringVar(&cfg.Addr, "addr", "localhost:8085", "Address on which to listen")
	flag.DurationVar(&cfg.CpuThinkTime, "cpu-think-time", 5*time.Second,
		"Time the computer has to think about a move")
	flag.IntVar(&cfg.CpuMaxFlags, "cpu-max-flags", 5,
		"Maximum flag moves to consider in any turn. <= 0 means unlimited")
	flag.StringVar(&cfg.TlsCertChain, "tls-cert", "", "Path to chain.pem for TLS")
	flag.StringVar(&cfg.TlsPrivKey, "tls-key", "", "Path to privkey.pem for TLS")
	flag.Parse()

	if len(flag.Args()) > 0 {
		fmt.Fprintf(os.Stderr, "unexpected extra arguments: %v\n", flag.Args())
		os.Exit(1)
	}

	hexz.NewCPUPlayerServer(cfg).Serve()
}
