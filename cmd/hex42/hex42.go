package main

import (
	"fmt"
	"math/rand"
	"strings"
)

func main() {
	// No matter how you place the pieces of two players,
	// on a 11x11 hex board you will always either have a path
	// from top to bottom by player 1, or from left to right by
	// player 2. Fascinating!
	const sideLen = 11
	var board [sideLen][sideLen]byte
	for i, r := range board {
		for j, _ := range r {
			board[i][j] = 'X'
			if rand.Intn(2) == 0 {
				board[i][j] = 'O'
			}
		}
	}

	for i, r := range board {
		fmt.Print(strings.Repeat(" ", i))
		for j, _ := range r {
			fmt.Printf("%c ", board[i][j])
		}
		fmt.Print("\n")
	}
	fmt.Print("\n")
	for i, r := range board {
		fmt.Print(strings.Repeat(" ", i))
		for j, _ := range r {
			if board[i][j] == 'X' {
				fmt.Printf("%c ", board[i][j])
			} else {
				fmt.Print("  ")
			}

		}
		fmt.Printf("\n")
	}
	fmt.Print("\n")
	for i, r := range board {
		fmt.Print(strings.Repeat(" ", i))
		for j, _ := range r {
			if board[i][j] == 'O' {
				fmt.Printf("%c ", board[i][j])
			} else {
				fmt.Print("  ")
			}

		}
		fmt.Printf("\n")
	}
	fmt.Print("\n")
}
