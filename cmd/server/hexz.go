package main

import (
	"flag"

	"github.com/dnswlt/hackz/hexz"
)

var ()

func main() {
	cfg := &hexz.ServerConfig{}

	flag.StringVar(&cfg.ServerAddress, "address", "", "Address on which to listen")
	flag.IntVar(&cfg.ServerPort, "port", 8084, "Port on which to listen")
	flag.StringVar(&cfg.DocumentRoot, "document-root", ".", "Root directory from which to serve files")
	flag.IntVar(&cfg.GameGcDelaySeconds, "gcdelay", 5,
		"Seconds to wait before deleting a disconnected player from a game")
	flag.StringVar(&cfg.TlsCertChain, "tls-cert", "", "Path to chain.pem for TLS")
	flag.StringVar(&cfg.TlsPrivKey, "tls-key", "", "Path to privkey.pem for TLS")
	flag.Parse()

	hexz.Serve(cfg)
}
