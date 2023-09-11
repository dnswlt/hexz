package hexz

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"strconv"
	"strings"
)

type HexzTestClient struct {
	serverURL string
	client    *http.Client
}

func newHexzTestClient(serverURL string) (*HexzTestClient, error) {
	jar, err := cookiejar.New(nil)
	if err != nil {
		return nil, err
	}
	return &HexzTestClient{
		serverURL: serverURL,
		client: &http.Client{
			Jar: jar, // Set cookie jar for passing along playerId
		},
	}, nil
}

// Logs in the given user name. Subsequent requests will send along the
// cookie obtained during login.
func (c *HexzTestClient) login(name string) error {
	loginForm := url.Values{}
	loginForm.Add("name", name)
	loginResp, err := c.client.PostForm(c.serverURL+"/hexz/login", loginForm)
	if err != nil {
		return err
	}
	if loginResp.StatusCode != http.StatusOK {
		return fmt.Errorf("could not log in: %s", loginResp.Status)
	}
	return nil
}

// newFlagzGame (somewhat unsurprisingly) starts a new Flagz game.
func (c *HexzTestClient) newFlagzGame(singlePlayer bool) (gameId string, err error) {
	newGameForm := url.Values{}
	newGameForm.Add("type", string(gameTypeFlagz))
	newGameForm.Add("singlePlayer", strconv.FormatBool(singlePlayer))
	newGameResp, err := c.client.PostForm(c.serverURL+"/hexz/new", newGameForm)
	if err != nil {
		return "", fmt.Errorf("could not start new game: %s", err)
	}
	if newGameResp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("could not start new game: %s", newGameResp.Status)
	}
	return gameIdFromPath(newGameResp.Request.URL.Path), nil
}

func (c *HexzTestClient) makeMove(gameId string, m *MoveRequest) error {
	data, _ := json.Marshal(m)
	resp, err := c.client.Post(c.serverURL+"/hexz/move/"+gameId, "application/json", bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("failed to make a move: %s", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to make a move: %s", resp.Status)
	}
	return nil
}

func (c *HexzTestClient) validMoves(gameId string) ([]*MoveRequest, error) {
	validMovesResp, err := c.client.Get(c.serverURL + "/hexz/moves/" + gameId)
	if err != nil {
		return nil, fmt.Errorf("cannot get valid moves: %s", err)
	}
	defer validMovesResp.Body.Close()
	if validMovesResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cannot get valid moves: %s", validMovesResp.Status)
	}
	dec := json.NewDecoder(validMovesResp.Body)
	var validMoves []*MoveRequest
	if err := dec.Decode(&validMoves); err != nil {
		return nil, fmt.Errorf("cannot unmarshal MoveRequest: %s", err)
	}
	return validMoves, nil
}

// Establishes a SSE connection with the given URL and sends the "data:" part
// of each event through the returned channel. Clients should close the done channel
// to abort the SSE connection.
func (c *HexzTestClient) receiveEvents(ctx context.Context, url string) (<-chan string, error) {
	ch := make(chan string)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Accept", "text/event-stream")
	req.Header.Add("Connection", "keep-alive")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("received status %s: %s", http.StatusText(resp.StatusCode), msg)
	}
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" || line[0] == ':' {
				// SSE separates messages by empty lines. Lines starting with a colon are ignored.
				continue
			}
			dataPrefix := "data: "
			if strings.HasPrefix(line, dataPrefix) {
				select {
				case ch <- line[len(dataPrefix):]:
					break // sending the event was successful
				case <-ctx.Done():
					return
				}
			}
			// Ignore all other lines for now.
		}
	}()
	return ch, nil
}
