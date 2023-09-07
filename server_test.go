package hexz

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestValidPlayerName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"abc", true},
		{"abc.def", true},
		{"abc_def-123", true},
		{"1digit", true},
		{"HANS", true},
		{"Mørän", true},
		{"Jérôme", true},
		{"Strüßenbähn", true},
		{"123", true},
		{"_letter-or.digit", true},
		{"ab", false},      // Too short
		{"jens$", false},   // Invalid character
		{"dw@best", false}, // Invalid character
		{"", false},
		{"verylongusernamesarenotallowedalright", false},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if got := isValidPlayerName(test.name); got != test.want {
				t.Errorf("unexpected result %t for name %s", got, test.name)
			}
		})
	}
}

func TestSha256HexDigest(t *testing.T) {
	got := sha256HexDigest("foo")
	want := "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
	if got != want {
		t.Errorf("Want: %q, got: %q", want, got)
	}
}

func serverConfigForTest(t *testing.T) *ServerConfig {
	historyRoot := t.TempDir()
	return &ServerConfig{
		ServerAddress:   "localhost",
		ServerPort:      8999,
		DocumentRoot:    "./resources",
		GameHistoryRoot: historyRoot,
		DebugMode:       true,
	}
}

const (
	testPlayerId   = "testId"
	testPlayerName = "tester"
)

func TestHandeNewGame(t *testing.T) {
	cfg := serverConfigForTest(t)
	s := NewServer(cfg)
	if !s.loginPlayer(testPlayerId, testPlayerName) {
		t.Errorf("Cannot log in test player")
	}
	w := httptest.NewRecorder()
	// Create request with login form parameters.
	form := url.Values{}
	form.Add("type", string(gameTypeFlagz))
	form.Add("singlePlayer", "true")
	r := httptest.NewRequest(http.MethodPost, "/hexz/new", strings.NewReader(form.Encode()))
	r.AddCookie(makePlayerCookie(testPlayerId, 10*time.Second))
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
	t.Errorf("Ongoing games: %d", len(s.ongoingGames))
}
