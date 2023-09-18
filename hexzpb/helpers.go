package hexzpb

func (s *GameState) PlayerNames() []string {
	names := make([]string, len(s.Players))
	for i, p := range s.Players {
		names[i] = p.Name
	}
	return names
}

func (s *GameState) PlayerNum(playerId string) int {
	for i, p := range s.Players {
		if p.Id == playerId {
			return i + 1
		}
	}
	return 0 // spectator
}
