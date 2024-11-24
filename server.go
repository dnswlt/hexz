package hexz

import (
	"crypto/sha256"
	"fmt"
)

// func (s *Server) handleView(w http.ResponseWriter, r *http.Request) {
// 	gameId := r.PathValue("gameId")
// 	if !isValidGameId(gameId) {
// 		http.Error(w, "Invalid game ID", http.StatusBadRequest)
// 		return
// 	}
// 	seqNum := r.PathValue("seqNum")
// 	if seqNum == "" {
// 		// No move number specified. Redirect to move 0.
// 		http.Redirect(w, r, fmt.Sprintf("%s/0", r.URL.Path), http.StatusSeeOther)
// 		return
// 	}
// 	s.serveHtmlFile(w, viewHtmlFilename)
// }

// func NewGameHistoryResponse(hist *GameHistory) *GameHistoryResponse {
// 	entries := make([]*GameHistoryResponseEntry, len(hist.Entries))
// 	for i, e := range hist.Entries {
// 		entries[i] = &GameHistoryResponseEntry{
// 			Timestamp:  e.Timestamp,
// 			EntryType:  e.EntryType,
// 			Move:       e.Move,
// 			Board:      e.Board,
// 			MoveScores: e.MoveScores,
// 		}
// 	}
// 	return &GameHistoryResponse{
// 		GameId:      hist.Header.GameId,
// 		PlayerNames: hist.Header.PlayerNames,
// 		GameType:    hist.Header.GameType,
// 		Entries:     entries,
// 	}
// }

// func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
// 	gameId := r.PathValue("gameId")
// 	if !isValidGameId(gameId) {
// 		http.Error(w, "Invalid game ID", http.StatusBadRequest)
// 		return
// 	}
// 	hist, err := s.readGameHistoryFromFile(gameId)
// 	if err != nil {
// 		http.Error(w, "", http.StatusNotFound)
// 		return
// 	}
// 	w.Header().Set("Content-Type", "application/json")
// 	var z io.Writer = w
// 	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
// 		w.Header().Set("Content-Encoding", "gzip")
// 		gz := gzip.NewWriter(w)
// 		defer gz.Close()
// 		z = gz
// 	}
// 	enc := json.NewEncoder(z)
// 	err = enc.Encode(NewGameHistoryResponse(hist))
// 	if err != nil {
// 		http.Error(w, "", http.StatusInternalServerError)
// 		hlog.Fatalf("Failed to marshal history response: %s", err)
// 	}
// }

func sha256HexDigest(pass string) string {
	passSha256Bytes := sha256.Sum256([]byte(pass))
	return fmt.Sprintf("%x", passSha256Bytes)
}

// func isLocalAddr(addr string) bool {
// 	host, _, err := net.SplitHostPort(addr)
// 	if err != nil {
// 		return false
// 	}
// 	ip := net.ParseIP(host)
// 	if ip == nil {
// 		return false
// 	}
// 	return ip.IsLoopback()
// }

// func (s *Server) basicAuthHandlerFunc(h http.HandlerFunc) http.HandlerFunc {
// 	return http.HandlerFunc(
// 		func(w http.ResponseWriter, r *http.Request) {
// 			if isLocalAddr(r.RemoteAddr) {
// 				// No authentication required
// 				s.IncCounter("/auth/granted/local")
// 				h(w, r)
// 				return
// 			}
// 			if s.config.AuthTokenSha256 == "" {
// 				// No auth token: only local access is allowed.
// 				s.IncCounter("/auth/rejected/nonlocal")
// 				http.Error(w, "", http.StatusForbidden)
// 				return
// 			}
// 			_, pass, ok := r.BasicAuth()
// 			passSha256 := sha256HexDigest(pass)
// 			rejected := true
// 			if !ok {
// 				s.IncCounter("/auth/rejected/missing_token")
// 			} else if passSha256 != s.config.AuthTokenSha256 {
// 				s.IncCounter("/auth/rejected/bad_passwd")
// 			} else {
// 				rejected = false
// 			}
// 			if rejected {
// 				w.Header().Set("WWW-Authenticate", `Basic realm="restricted", charset="UTF-8"`)
// 				http.Error(w, "", http.StatusUnauthorized)
// 				return
// 			}
// 			s.IncCounter("/auth/granted/basic_auth")
// 			h(w, r)
// 		})
// }
