package hexz

import (
	"crypto/sha256"
	"fmt"
)

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
