package main

import (
	"flag"
	"time"

	"github.com/dnswlt/hackz/hexz"
)

var ()

func main() {
	cfg := &hexz.ServerConfig{}

	flag.StringVar(&cfg.ServerAddress, "address", "", "Address on which to listen")
	flag.IntVar(&cfg.ServerPort, "port", 8084, "Port on which to listen")
	flag.StringVar(&cfg.DocumentRoot, "document-root", ".", "Root directory from which to serve files")
	flag.DurationVar(&cfg.PlayerRemoveDelay, "remove-delay", time.Duration(60)*time.Second,
		"Seconds to wait before removing a disconnected player from a game")
	flag.DurationVar(&cfg.LoginTtl, "login-ttl", time.Duration(24)*time.Hour,
		"Seconds to wait before deleting a disconnected player from a game")
	flag.BoolVar(&cfg.DebugMode, "debug", false,
		"Run server in debug mode. Only set to true during development.")
	flag.StringVar(&cfg.TlsCertChain, "tls-cert", "", "Path to chain.pem for TLS")
	flag.StringVar(&cfg.TlsPrivKey, "tls-key", "", "Path to privkey.pem for TLS")
	flag.Parse()

	hexz.Serve(cfg)
}
