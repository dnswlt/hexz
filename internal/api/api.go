// Package api contains API messages that are used by multiple
// internal packages and hence need to live in a leaf package
// to avoid circular imports.
//
// TODO: consider moving all API stuff here, or make it protobuf based.
package api

import "time"

type GameType string

// A random UUID used to identify players. Also used in cookies.
type PlayerId string

// Player has JSON annotations for serialization to disk.
// It is not used in the public API.
type Player struct {
	Id         PlayerId  `json:"id"`
	Name       string    `json:"name"`
	LastActive time.Time `json:"lastActive"`
}

// BoardStatus contains summary information about the status of a game's board.
// It is a convenience type that holds commonly used info in one struct.
type BoardStatus struct {
	Done  bool  `json:"done"`
	Score []int `json:"score"`
	Move  int   `json:"move"`
	Turn  int   `json:"turn"`
}

// Used to report CPU stats by clients.
type WASMStatsRequest struct {
	GameId   string    `json:"gameId"`
	GameType GameType  `json:"gameType"`
	Move     int       `json:"move"`
	UserInfo UserInfo  `json:"userInfo"`
	Stats    WASMStats `json:"stats"`
}

type WASMStats struct {
	// MCTS stats.
	TreeSize   int           `json:"treeSize"`
	MaxDepth   int           `json:"maxDepth"`
	Iterations int           `json:"iterations"`
	Elapsed    time.Duration `json:"elapsed"`
	// Memory allocations, in MiB (1024*1024 bytes).
	TotalAllocMiB float64 `json:"totalAllocMiB"`
	HeapAllocMiB  float64 `json:"heapAllocMiB"`
}

type UserInfo struct {
	// The User-Agent header.
	UserAgent string `json:"userAgent"`
	// Taken from navigator.language.
	Language string `json:"language"`
	// Resolution is the screen resolution in pixels [window.screen.width, window.screen.height].
	Resolution [2]int `json:"resolution"`
	// Viewport is the size of the viewport in pixels [window.innerWidth, window.innerHeight].
	Viewport [2]int `json:"viewport"`
	// BrowserWindow is the size of the browser window in pixels [window.outerWidth, window.outerHeight].
	BrowserWindow [2]int `json:"browserWindow"`
	// HardwareConcurrency is the number of logical processors available to run threads
	// on the user's computer (navigator.hardwareConcurrency).
	HardwareConcurrency int `json:"hardwareConcurrency"`
}
