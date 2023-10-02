package main

import (
	"context"
	"flag"
	"os"
	"regexp"
	"strconv"
	"time"

	"github.com/dnswlt/hexz"
	"github.com/dnswlt/hexz/hexzmem"
	"github.com/dnswlt/hexz/hexzsql"
	"github.com/dnswlt/hexz/hlog"
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
	flag.StringVar(&cfg.GameHistoryRoot, "history-dir", "",
		"Root directory in whicih to read/write history files. If empty, history is disabled.")
	flag.StringVar(&cfg.LoginDatabasePath, "userdb", "_logins.json",
		"File in which to store login information if the local in-memory login store is used.")
	flag.StringVar(&cfg.RedisAddr, "redis-addr", "",
		"Address of the Redis server. Only used by the -stateless server (default: localhost:6379).")
	flag.StringVar(&cfg.PostgresURL, "postgres-url", "",
		"URL of the PostgreSQL server (e.g. \"postgres://hexz:hexz@localhost:5432/hexz\"). If empty, no persistent storage is used.")
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
	flag.BoolVar(&cfg.DisableUndo, "disable-undo", false, "If true, games will not support undo/redo")
	flag.BoolVar(&cfg.Stateless, "stateless", false, "If true, run in stateless mode (e.g. Cloud Run)")
	flag.Parse()
	setFlags := make(map[string]bool)
	flag.Visit(func(f *flag.Flag) {
		setFlags[f.Name] = true
	})
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
	if cfg.Stateless {
		// Redis
		if cfg.RedisAddr == "" {
			cfg.RedisAddr = "localhost:6379"
		}
		rc, err := hexzmem.NewRedisClient(&hexzmem.RedisClientConfig{
			Addr:     cfg.RedisAddr,
			LoginTTL: cfg.LoginTTL,
			GameTTL:  cfg.InactivityTimeout,
		})
		if err != nil {
			hlog.Fatalf("error connecting to redis: %s", err)
		}
		hlog.Infof("connected to Redis at %s", cfg.RedisAddr)
		// Postgres (optional)
		var dbStore *hexzsql.PostgresStore
		if cfg.PostgresURL != "" {
			dbStore, err = hexzsql.NewPostgresStore(context.Background(), cfg.PostgresURL)
			if err != nil {
				hlog.Fatalf("error connecting to postgres: %s", err)
			}
			hlog.Infof("connected to PostgreSQL at %s", redactPGPassword(cfg.PostgresURL))
		}
		// Let's go!
		s, err := hexz.NewStatelessServer(cfg, &hexzmem.RemotePlayerStore{RedisClient: rc}, rc, dbStore)
		if err != nil {
			hlog.Fatalf("error creating server: %s", err)
		}
		s.Serve()
		return // never reached.
	}

	// Stateful server.
	s, err := hexz.NewServer(cfg)
	if err != nil {
		hlog.Fatalf("error creating server: %v\n", err)
	}
	s.Serve()
}
