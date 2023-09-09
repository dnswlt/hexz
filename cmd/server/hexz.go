package main

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"time"

	"github.com/dnswlt/hexz"
)

var ()

func main() {
	cfg := &hexz.ServerConfig{}

	flag.StringVar(&cfg.ServerAddress, "address", "", "Address on which to listen")
	flag.IntVar(&cfg.ServerPort, "port", 8084, "Port on which to listen")
	flag.StringVar(&cfg.DocumentRoot, "resources-dir", "./resources",
		"Root directory from which to serve files")
	flag.StringVar(&cfg.GameHistoryRoot, "history-dir", "./hist",
		"Root directory from which to read game history files")
	flag.DurationVar(&cfg.InactivityTimeout, "inactivity-timeout", 60*time.Minute,
		"Time to wait before ending a game due to inactivity")
	flag.DurationVar(&cfg.PlayerRemoveDelay, "remove-delay", 60*time.Second,
		"Time to wait before removing a disconnected player from a game")
	flag.DurationVar(&cfg.LoginTtl, "login-ttl", 24*time.Hour,
		"Time to wait logging a player out after inactivity")
	flag.DurationVar(&cfg.CpuThinkTime, "cpu-think-time", 5*time.Second,
		"Time the computer has to think about a move")
	flag.IntVar(&cfg.CpuMaxFlags, "cpu-max-flags", 5,
		"Maximum flag moves to consider in any turn. <= 0 means unlimited")
	flag.BoolVar(&cfg.DebugMode, "debug", false,
		"Run server in debug mode. Only set to true during development.")
	flag.StringVar(&cfg.AuthTokenSha256, "auth-token", "", "SHA256 token for access to restricted paths (http authentication)")
	flag.StringVar(&cfg.TlsCertChain, "tls-cert", "", "Path to chain.pem for TLS")
	flag.StringVar(&cfg.TlsPrivKey, "tls-key", "", "Path to privkey.pem for TLS")
	flag.BoolVar(&cfg.EnableUndo, "enable-undo", true, "If true, games support undo/redo")
	flag.Parse()

	if cfg.AuthTokenSha256 != "" {
		if len(cfg.AuthTokenSha256) != 64 || !regexp.MustCompile("[a-fA-F0-9]+").MatchString(cfg.AuthTokenSha256) {
			fmt.Fprint(os.Stderr, "-auth-token must be a SHA256 hex digest")
			os.Exit(1)
		}
	}
	hexz.NewServer(cfg).Serve()
}
