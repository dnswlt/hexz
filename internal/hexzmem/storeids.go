package hexzmem

import "github.com/dnswlt/hexz/internal/xrand"

const (
	uppercaseLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	hexDigits        = "0123456789abcdef"
)

func generateRandomID(n int, characters string) string {
	id := make([]byte, n)
	l := len(characters)
	for i := 0; i < n; i++ {
		j := xrand.Intn(l)
		id[i] = characters[j]
	}
	return string(id)
}

// Generates a game ID consisting of 6 uppercase ASCII letters.
func GenerateGameID() string {
	return generateRandomID(6, uppercaseLetters)
}

func GeneratePubsubID() string {
	return generateRandomID(16, hexDigits)
}
