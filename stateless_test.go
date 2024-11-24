package hexz

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"
)

func newTestStatelessServer(config *ServerConfig) (*StatelessServer, error) {
	renderer, err := NewRenderer()
	if err != nil {
		return nil, err
	}
	playerStore, err := NewInMemoryPlayerStore(config.LoginTTL, config.LoginDatabasePath)
	if err != nil {
		return nil, err
	}
	gameStore := NewInMemoryGameStore()
	b := NewStatelessServerBuilder(config, playerStore, gameStore, renderer)
	return b.Build(), nil
}

func TestHandleNewGame(t *testing.T) {
	cfg := serverConfigForTest(t)
	s, err := newTestStatelessServer(cfg)
	if err != nil {
		t.Fatal("Could not create server:", err)
	}
	if err := s.playerStore.Login(context.Background(), testPlayerId, testPlayerName); err != nil {
		t.Error("Cannot log in test player: ", err)
	}
	w := httptest.NewRecorder()
	// Create request with login form parameters.
	form := url.Values{}
	form.Add("type", string(gameTypeFlagz))
	form.Add("singlePlayer", "true")
	r := httptest.NewRequest(http.MethodPost, "/hexz/new", strings.NewReader(form.Encode()))
	r.AddCookie(s.makePlayerCookie(testPlayerId, 24*time.Hour))
	r.Header.Add("Content-Type", "application/x-www-form-urlencoded")

	s.handleNewGame(w, r)

	// Expect a redirect to /hexz/{gameId}
	resp := w.Result()
	want := http.StatusSeeOther
	if resp.StatusCode != want {
		msg, _ := io.ReadAll(resp.Body)
		t.Errorf("Want: %s, got: %s %q", http.StatusText(want), resp.Status, msg)
	}
	loc := resp.Header.Get("Location")
	if pattern := `/hexz/[A-Z]{6}`; !regexp.MustCompile(pattern).MatchString(loc) {
		t.Errorf("Wrong Location header: want: %s, got: %q", pattern, loc)
	}
	recentGames, err := s.gameStore.ListRecentGames(context.Background(), 100)
	if err != nil {
		t.Fatal("Could not get recent games:", err)
	}
	if len(recentGames) != 1 {
		t.Errorf("Ongoing games: %d, want: 1", len(recentGames))
	}
}

func TestURLJoinPath(t *testing.T) {
	tests := []struct {
		prefix string
		suffix string
		want   string
	}{
		{"", "", ""},
		{"", "/foo", "/foo"},
		{"", "foo", "foo"},
		{"/foo", "", "/foo"},
		{"foo", "", "foo"},
		{"/foo/", "/bar", "/foo/bar"},
		{"/foo/", "/bar/", "/foo/bar/"},
		{"/foo", "", "/foo"},
		{"/foo", "/", "/foo/"},
		{"", "/bar", "/bar"},
	}
	for _, tc := range tests {
		if got := urlJoinPath(tc.prefix, tc.suffix); got != tc.want {
			t.Errorf("Invalid prefix: got %q, want %q", got, tc.want)
		}
	}
}
