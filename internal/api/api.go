// Package api contains API messages that are used by multiple
// internal packages and hence need to live in a leaf package
// to avoid circular imports.
//
// TODO: consider moving all API stuff here, or make it protobuf based.
package api

type GameType string

// A random UUID used to identify players. Also used in cookies.
type PlayerId string

// BoardStatus contains summary information about the status of a game's board.
// It is a convenience type that holds commonly used info in one struct.
type BoardStatus struct {
	Done  bool  `json:"done"`
	Score []int `json:"score"`
	Move  int   `json:"move"`
	Turn  int   `json:"turn"`
}
