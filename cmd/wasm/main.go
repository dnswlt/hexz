//go:build wasm

package main

import (
	"encoding/json"
	"fmt"
	"runtime"
	"syscall/js"
	"time"

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
	"google.golang.org/protobuf/proto"
)

type suggestMoveArgs struct {
	EncodedGameState   []byte `json:"encodedGameState"`
	MaxThinkTimeMillis int    `json:"maxThinkTimeMillis"`
}

type suggestMoveResult struct {
	MoveRequest hexz.MoveRequest `json:"moveRequest"`
	Stats       hexz.WASMStats   `json:"stats"`
}

func main() {
	fmt.Printf("This line was written by the goWASM Heavy Metal Superengine!!!\n")
	goWasmSuggestMove := js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) != 1 || args[0].Type() != js.TypeString {
			fmt.Printf("goWasmSuggestMove: called with invalid arguments [%v]%v\n", this, args)
			return nil
		}
		var a suggestMoveArgs
		err := json.Unmarshal([]byte(args[0].String()), &a)
		if err != nil {
			fmt.Printf("goWasmSuggestMove: called with invalid JSON: %s\n", err)
			return nil
		}
		gameState := &pb.GameState{}
		var ge hexz.GameEngine
		err = proto.Unmarshal(a.EncodedGameState, gameState)
		if err != nil {
			fmt.Printf("goWasmSuggestMove: called with invalid encoded proto: %s\n", err)
			return nil
		}
		ge, err = hexz.DecodeGameEngine(gameState.EngineState)
		if err != nil {
			fmt.Printf("goWasmSuggestMove: cannot decode game engine: %s\n", err)
			return nil
		}
		if ge.IsDone() {
			fmt.Printf("goWasmSuggestMove: game is already done\n")
			return nil
		}
		spge, ok := ge.(*hexz.GameEngineFlagz)
		if !ok {
			fmt.Printf("goWasmSuggestMove: %T is not supported, only GameEngineFlagz is\n", ge)
			return nil
		}
		mcts := hexz.NewMCTS()
		maxThinkTime := time.Duration(a.MaxThinkTimeMillis) * time.Millisecond
		mv, stats := mcts.SuggestMove(spge, maxThinkTime, 0)
		var memstats runtime.MemStats
		runtime.ReadMemStats(&memstats)
		heapAllocMiB := float64(memstats.HeapAlloc) / (1024 * 1024)
		totalAllocMiB := float64(memstats.TotalAlloc) / (1024 * 1024)
		res, err := json.Marshal(suggestMoveResult{
			MoveRequest: hexz.MoveRequest{
				Move: mv.Move,
				Row:  mv.Row,
				Col:  mv.Col,
				Type: mv.CellType,
			},
			Stats: hexz.WASMStats{
				TreeSize:      stats.TreeSize,
				MaxDepth:      stats.MaxDepth,
				Iterations:    stats.Iterations,
				Elapsed:       stats.Elapsed,
				HeapAllocMiB:  heapAllocMiB,
				TotalAllocMiB: totalAllocMiB,
			},
		})
		if err != nil {
			panic("Cannot marshal result: " + err.Error())
		}
		return string(res)
	})
	js.Global().Set("goWasmSuggestMove", goWasmSuggestMove)

	<-make(chan bool) // Run forever, else we'll lose access to all exported functions, too.
	goWasmSuggestMove.Release()
}
