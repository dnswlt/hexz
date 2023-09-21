package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"
)

type jsArg struct {
	A string `json:"a"`
	B int    `json:"b"`
}

func main() {
	fmt.Printf("This line was written by the goWASM Heavy Metal Superengine!!!\n")
	goWasmFoo := js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) != 1 || args[0].Type() != js.TypeString {
			fmt.Printf("goWasmFoo was called with invalid arguments [%v]%v\n", this, args)
			return false
		}
		var m jsArg
		if err := json.Unmarshal([]byte(args[0].String()), &m); err != nil {
			fmt.Printf("goWasmFoo was called with invalid JSON: %v\n", args)
			return false
		}
		fmt.Printf("goWasmFoo was called with valid JSON. Happy birthday! %v\n", m)
		return true
	})
	js.Global().Set("goWasmFoo", goWasmFoo)
	<-make(chan bool)
	goWasmFoo.Release()
}
