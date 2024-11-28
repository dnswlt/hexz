package main

import (
	"context"
	"errors"
	"flag"
	"os"
	"path"
	"regexp"
	"strconv"
	"time"

	"github.com/dnswlt/hexz/internal/hexzmem"
	"github.com/dnswlt/hexz/internal/hexzsql"
	"github.com/dnswlt/hexz/internal/hlog"
	"github.com/dnswlt/hexz/pkg/hexz"
	"github.com/dnswlt/hexz/pkg/hexzpb"
)

func redactPGPassword(url string) string {
	reURL := regexp.MustCompile(`(?i)(postgres://[^:]+:)[^@]+(@)`)
	reDSN := regexp.MustCompile(`(?i)(password=)[^ ]+`)
	url = reDSN.ReplaceAllString(url, "$1<redacted>")
	return reURL.ReplaceAllString(url, "$1<redacted>$2")
}

func main() {
	cfg := &hexz.ServerConfig{}

	flag.StringVar(&cfg.ServerHost, "host", "", "Hostname/IP on which to listen. Leave empty to listen on all interfaces.")
	flag.IntVar(&cfg.ServerPort, "port", 8080, "Port on which to listen")
	flag.StringVar(&cfg.DocumentRoot, "resources-dir", "./resources",
		"Root directory from which to serve files")
	flag.StringVar(&cfg.URLPathPrefix, "url-path-prefix", "/hexz", "Path prefix for all server URLs")
	flag.StringVar(&cfg.GameHistoryRoot, "history-dir", "",
		"Root directory in whicih to read/write history files. If empty, history is disabled.")
	flag.StringVar(&cfg.LoginDatabasePath, "userdb", "_logins.json",
		"File in which to store login information if the local in-memory login store is used.")
	flag.StringVar(&cfg.RemoteCPUPlayerURL, "remote-cpu-url", "",
		"Base URL of the CPU player server. If empty, the in-process or WASM CPU engine is used.")
	flag.StringVar(&cfg.RedisAddr, "redis-addr", "",
		"Address of the Redis server storing game states. If empty, an embedded in-memory store is used.")
	flag.StringVar(&cfg.PostgresURL, "postgres-url", "",
		"URL of the PostgreSQL server (e.g. \"postgres://hexz:hexz@localhost:5432/hexz\"). If empty, no persistent storage is used.")
	cpuPlayerMode := flag.String("cpu-player-mode", "embedded", "Mode in which to run CPU players. One of {wasm, embedded, remote}")
	flag.DurationVar(&cfg.InactivityTimeout, "inactivity-timeout", 60*time.Minute,
		"Time to wait before ending a game due to inactivity")
	flag.DurationVar(&cfg.PlayerRemoveDelay, "remove-delay", 60*time.Second,
		"Time to wait before removing a disconnected player from a game")
	flag.DurationVar(&cfg.LoginTTL, "login-ttl", 24*time.Hour,
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
	logFormat := flag.String("log-format", "plain", "Format of log messages. One of {plain, json}.")
	flag.Parse()
	setFlags := make(map[string]bool)
	flag.Visit(func(f *flag.Flag) {
		setFlags[f.Name] = true
	})
	switch *cpuPlayerMode {
	case "wasm":
		wasmFile := path.Join(cfg.DocumentRoot, "wasm", "hexz.wasm.gz")
		if _, err := os.Stat(wasmFile); errors.Is(err, os.ErrNotExist) {
			hlog.Fatalf("WASM file %s not found", wasmFile)
		}
		cfg.CPUPlayerMode = hexzpb.CPUPlayerMode_WASM
	case "embedded":
		if cfg.RemoteCPUPlayerURL != "" {
			hlog.Fatalf("-cpu-player-mode=embedded and -remote-cpu-url are mutually exclusive")
		}
		cfg.CPUPlayerMode = hexzpb.CPUPlayerMode_EMBEDDED_CPU
	case "remote":
		if cfg.RemoteCPUPlayerURL == "" {
			hlog.Fatalf("-cpu-player-mode=remote requires -remote-cpu-url to be set")
		}
		cfg.CPUPlayerMode = hexzpb.CPUPlayerMode_REMOTE_CPU
	default:
		hlog.Fatalf("Invalid value for -cpu-player-mode: %s", *cpuPlayerMode)
	}
	// If -port was not specified explicitly, try the $PORT environment variable.
	envPort := os.Getenv("PORT")
	if !setFlags["port"] && envPort != "" {
		port, err := strconv.Atoi(envPort)
		if err != nil {
			hlog.Infof("invalid port: %v\n", envPort)
		}
		cfg.ServerPort = port
	}
	// If -redis-addr was not specified explicitly, try the $REDISHOST and $REDISPORT environment variables.
	envRedisHost := os.Getenv("REDISHOST")
	envRedisPort := os.Getenv("REDISPORT")
	if !setFlags["redis-addr"] && envRedisHost != "" && envRedisPort != "" {
		cfg.RedisAddr = envRedisHost + ":" + envRedisPort
	}
	// If -postgres-url was not specified explicitly, try the $POSTGRES_URL environment variable.
	envPostgresURL := os.Getenv("POSTGRES_URL")
	if !setFlags["postgres-url"] && envPostgresURL != "" {
		cfg.PostgresURL = envPostgresURL
	}
	if cfg.AuthTokenSha256 != "" {
		if len(cfg.AuthTokenSha256) != 64 || !regexp.MustCompile("[a-fA-F0-9]+").MatchString(cfg.AuthTokenSha256) {
			hlog.Fatalf("-auth-token must be a SHA256 hex digest")
		}
	}
	if len(flag.Args()) > 0 {
		hlog.Fatalf("unexpected extra arguments: %v", flag.Args())
	}
	if *logFormat == "json" {
		hlog.UseJSONLogger()
	}
	// Build a stateless server
	renderer, err := hexz.NewRenderer(path.Join(cfg.DocumentRoot, "templates"))
	if err != nil {
		hlog.Fatalf("error creating renderer: %v", err)
	}
	renderer.SetAutoReload(cfg.DebugMode)

	var gameStore hexzmem.GameStore
	var playerStore hexzmem.PlayerStore
	if cfg.RedisAddr != "" {
		rc, err := hexzmem.NewRedisClient(&hexzmem.RedisClientConfig{
			Addr:     cfg.RedisAddr,
			LoginTTL: cfg.LoginTTL,
			GameTTL:  cfg.InactivityTimeout,
		})
		if err != nil {
			hlog.Fatalf("error connecting to redis: %s", err)
		}
		hlog.Infof("connected to Redis at %s", cfg.RedisAddr)
		//Redis stores
		playerStore = &hexzmem.RemotePlayerStore{RedisClient: rc}
		gameStore = rc
	} else {
		// Local stores
		hlog.Infof("Using in memory player and game stores. Login DB: %s", cfg.LoginDatabasePath)
		playerStore, err = hexzmem.NewInMemoryPlayerStore(cfg.LoginTTL, cfg.LoginDatabasePath)
		if err != nil {
			hlog.Fatalf("error creating in-memory player store: %v", err)
		}
		gameStore = hexzmem.NewInMemoryGameStore(cfg.InactivityTimeout)
	}
	b := hexz.NewStatelessServerBuilder(cfg, playerStore, gameStore, renderer)
	// Postgres (optional)
	if cfg.PostgresURL != "" {
		var dbStore hexzsql.DatabaseStore
		dbStore, err := hexzsql.NewPostgresStore(context.Background(), cfg.PostgresURL)
		if err != nil {
			hlog.Fatalf("error connecting to postgres: %s", err)
		}
		hlog.Infof("connected to PostgreSQL at %s", redactPGPassword(cfg.PostgresURL))
		b = b.WithDatabaseStore(dbStore)
	}
	// Remote CPU (optional)
	if cfg.RemoteCPUPlayerURL != "" {
		client, err := hexz.NewCPUPlayerServiceClient(cfg.RemoteCPUPlayerURL)
		if err != nil {
			hlog.Fatalf("error connecting to remote CPU player: %v", err)
		}
		b = b.WithCPUPlayerServiceClient(client)
	}
	s := b.Build()
	s.Serve()
}
