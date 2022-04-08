//static evaluation of a board state - checks wins, losses
//two full and two empties, and 3 full and 1 empty for every
//4-space window on the board
func stateHeuristic(board [][]int, player int) int {
	score := 0

	checkWindow := func(window []int, player int) int {
		score, playerCount, otherPlayerCount, emptyCount := 0, 0, 0, 0
		for i := 0; i < len(window); i++ {
			if window[i] == player {
				playerCount++
			} else if window[i] == BLANK {
				emptyCount++
			} else {
				otherPlayerCount++
			}
		}
		if playerCount == 4 {
			score = 100
		} else if playerCount == 3 && emptyCount == 1 {
			score = 5
		} else if playerCount == 2 && emptyCount == 2 {
			score = 2
		}
		if otherPlayerCount == 3 && emptyCount == 1 {
			score -= 4
		}
		return score
	}

	//score horizontal
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i])-3; j++ {
			window := make([]int, 4)
			for k := j; k < j+4; k++ {
				window[k-j] = board[i][k]
			}
			score += checkWindow(window, player)
		}
	}

	//score vertical
	for j := 0; j < len(board[0]); j++ {
		for i := 0; i < len(board)-3; i++ {
			window := make([]int, 4)
			for k := i; k < i+4; k++ {
				window[k-i] = board[k][j]
			}
			score += checkWindow(window, player)
		}
	}

	//score diagonals
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {

			//check positive sloped line
			window := make([]int, 0)
			for diagI, diagJ := i, j; diagI >= 0 && diagJ < len(board[0]); diagI, diagJ = diagI-1, diagJ+1 {
				window = append(window, board[diagI][diagJ])
			}
			if len(window) == 4 {
				score += checkWindow(window, player)
			}
			window = make([]int, 0)
			for diagI, diagJ := i, j; diagI < len(board) && diagJ < len(board[0]); diagI, diagJ = diagI+1, diagJ+1 {
				window = append(window, board[diagI][diagJ])
			}
			if len(window) == 4 {
				score += checkWindow(window, player)
			}
		}
	}

	return score
}
