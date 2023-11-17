package hexz

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
)

// Export boards as SVG files for debugging.

// Helper struct to store evaluations per cell type.
type cellEvalScores struct {
	normal float32
	flag   float32
}

func (c *cellEvalScores) Max() float32 {
	if c == nil {
		return 0.0
	}
	if c.normal > c.flag {
		return c.normal
	}
	return c.flag
}

func ScaleRGB(col1 string, col2 string, scale float64) (string, error) {
	if col1[0] != '#' || col2[0] != '#' || len(col1) != 7 || len(col2) != 7 {
		return "", fmt.Errorf("only 6 digit hex colors are supported (#123456)")
	}
	a1, err := strconv.ParseInt(col1[1:7], 16, 64)
	if err != nil {
		return "", fmt.Errorf("ScaleRGB: parse col1: %w", err)
	}
	a2, err := strconv.ParseInt(col2[1:], 16, 64)
	if err != nil {
		return "", fmt.Errorf("ScaleRGB: parse col2: %w", err)
	}
	r1 := (a1 >> 16) & 0xff
	g1 := (a1 >> 8) & 0xff
	b1 := a1 & 0xff
	r2 := (a2 >> 16) & 0xff
	g2 := (a2 >> 8) & 0xff
	b2 := a2 & 0xff
	r := math.Round(float64(r1) + float64(r2-r1)*scale)
	g := math.Round(float64(g1) + float64(g2-g1)*scale)
	b := math.Round(float64(b1) + float64(b2-b1)*scale)
	if r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255 {
		return "", fmt.Errorf("outside the RGB range: %f %f %f", r, g, b)
	}
	return fmt.Sprintf("#%02x%02x%02x", int(r), int(g), int(b)), nil
}

// hexPolySVG returns an SVG <g> element representing the hex cell (r, c).
func hexPolySVG(sideLength float64, f *Field, r, c int, move *GameEngineMove, es *cellEvalScores) string {
	// Unscaled SVG paths for cell icons.
	cellIconPaths := map[CellType]string{
		cellFire:  "M -0.8562 -49.9998 L 3.7186 -44.2813 C 9.1362 -37.509 11.1328 -30.8269 10.9476 -24.416 C 10.7676 -18.1826 8.5309 -12.6255 6.2976 -8.0391 C 5.5471 -6.4981 4.7362 -4.95 3.9933 -3.5319 C 3.649 -2.8743 3.319 -2.2448 3.0167 -1.6567 C 2.0067 0.3076 1.2348 1.9333 0.7443 3.3595 C 0.2514 4.7914 0.16 5.6971 0.2229 6.2695 C 0.27 6.7014 0.4152 7.1362 0.9862 7.7071 C 1.9305 8.6514 2.6271 8.9167 3.0495 8.9976 C 3.4809 9.0809 4.0548 9.0538 4.8757 8.7167 C 6.7119 7.9614 8.9538 5.9952 11.3062 3.0457 C 13.5657 0.2124 15.5386 -3.0286 16.9686 -5.6219 C 17.6762 -6.9057 18.2357 -8.0025 18.6147 -8.7717 C 18.8038 -9.1558 18.9476 -9.4567 19.0419 -9.6566 L 19.1452 -9.8781 L 19.1676 -9.9277 L 19.1709 -9.9341 L 19.1714 -9.9357 L 19.1719 -9.9364 L 19.1719 -9.9366 L 21.9605 -16.1081 L 26.8247 -11.3921 C 37.8381 -0.7143 41.2461 14.7043 34.0071 28.7024 C 27.7119 40.8757 14.8205 49.1695 0 49.1695 C -20.9286 49.1695 -38.0952 32.5847 -38.0952 11.8824 C -38.0952 0.9757 -32.0625 -6.8286 -25.4918 -14.5012 C -24.6346 -15.5022 -23.7585 -16.5109 -22.8687 -17.5355 C -16.7675 -24.5605 -10.0167 -32.3334 -4.2276 -43.4985 L -0.8562 -49.9998 Z M 24.4138 0.5619 C 22.9276 3.1319 20.9947 6.1719 18.7519 8.9838 C 16.1195 12.2847 12.6324 15.8243 8.4981 17.5247 C 6.3333 18.4147 3.8614 18.8528 1.2495 18.35 C -1.3719 17.8452 -3.7162 16.4733 -5.7481 14.4414 C -7.8014 12.3881 -8.9543 9.9619 -9.2448 7.3038 C -9.52 4.7857 -8.9876 2.371 -8.2614 0.26 C -7.5328 -1.8562 -6.4895 -3.9957 -5.4529 -6.0114 C -5.0824 -6.7324 -4.7167 -7.4305 -4.3543 -8.1216 C -3.6486 -9.4677 -2.9562 -10.7892 -2.2652 -12.2084 C -0.21 -16.4289 1.3071 -20.5108 1.4276 -24.691 C 1.4924 -26.933 1.1581 -29.3329 0.13 -31.9123 C -5.1862 -23.3496 -10.8945 -16.7869 -15.7256 -11.2326 C -16.6011 -10.226 -17.4478 -9.2526 -18.2581 -8.3064 C -24.8819 -0.5719 -28.5714 4.8252 -28.5714 11.8824 C -28.5714 27.1062 -15.8904 39.6457 0 39.6457 C 11.2109 39.6457 20.8657 33.3819 25.5476 24.3276 C 29.6509 16.3928 29.0709 7.8014 24.4138 0.5619 Z",
		cellFlag:  "M 3.3072 -38.5457 C -5.0739 -45.2506 -14.9125 -45.0416 -21.9362 -43.5629 C -26.2188 -42.6613 -30.5708 -41.1876 -34.3913 -39.0083 C -36.0409 -38.0657 -37.0588 -36.3116 -37.0588 -34.4117 V 39.7059 C -37.0588 42.6298 -34.6885 45 -31.7647 45 C -28.8409 45 -26.4706 42.6298 -26.4706 39.7059 V 11.2733 C -19.185 8.2022 -9.9318 6.7759 -3.3072 12.0754 C 5.0739 18.7798 14.9125 18.5712 21.9362 17.0926 C 27.2959 15.9639 31.4688 14.0405 33.3286 13.1014 C 35.4012 12.0552 37.0588 10.4474 37.0588 7.9412 V -34.4117 C 37.0588 -36.299 36.054 -38.0434 34.4218 -38.9907 C 32.7907 -39.9373 30.7821 -39.9461 29.1441 -39.0116 L 29.1393 -39.0091 C 21.5264 -34.8494 10.5951 -32.7153 3.3072 -38.5457 Z M -26.4706 -31.0797 V -0.0053 C -16.5104 -3.1934 -5.1829 -2.9848 3.3072 3.807 C 8.1614 7.6908 14.2052 7.8999 19.755 6.7315 C 22.4132 6.1714 24.7601 5.3301 26.4706 4.6091 V -26.465 C 25.0888 -26.0227 23.5646 -25.6034 21.9362 -25.2606 C 14.9125 -23.7819 5.0739 -23.5729 -3.3072 -30.2777 C -9.9318 -35.5772 -19.185 -34.1507 -26.4706 -31.0797 Z",
		cellPest:  "M -4.5455 -43.1818 C -4.5455 -39.4162 -7.5982 -36.3636 -11.3636 -36.3636 C -15.1292 -36.3636 -18.1818 -39.4162 -18.1818 -43.1818 C -18.1818 -46.9474 -15.1292 -50 -11.3636 -50 C -7.5982 -50 -4.5455 -46.9474 -4.5455 -43.1818 Z M -2.2727 -18.1818 C -2.2727 -15.0438 -4.8164 -12.5 -7.9545 -12.5 C -11.0927 -12.5 -13.6364 -15.0438 -13.6364 -18.1818 C -13.6364 -21.3198 -11.0927 -23.8636 -7.9545 -23.8636 C -4.8164 -23.8636 -2.2727 -21.3198 -2.2727 -18.1818 Z M 4.5455 -26.1364 C 7.6836 -26.1364 10.2273 -28.6802 10.2273 -31.8182 C 10.2273 -34.9562 7.6836 -37.5 4.5455 -37.5 C 1.4073 -37.5 -1.1364 -34.9562 -1.1364 -31.8182 C -1.1364 -28.6802 1.4073 -26.1364 4.5455 -26.1364 Z M 5.6818 18.1818 C 5.6818 21.32 3.1382 23.8636 0 23.8636 C -3.1382 23.8636 -5.6818 21.32 -5.6818 18.1818 C -5.6818 15.0436 -3.1382 12.5 0 12.5 C 3.1382 12.5 5.6818 15.0436 5.6818 18.1818 Z M -27.2727 -36.3636 C -29.7831 -36.3636 -31.8182 -34.3285 -31.8182 -31.8182 C -31.8182 -29.3078 -29.7831 -27.2727 -27.2727 -27.2727 V 22.7273 C -27.2727 28.5323 -25.5955 35.2159 -21.6166 40.5682 C -17.5166 46.0836 -11.0809 50 -2.2727 50 C 6.5355 50 12.9714 46.0836 17.0714 40.5682 C 21.05 35.2159 22.7273 28.5323 22.7273 22.7273 V -27.2727 C 25.2377 -27.2727 27.2727 -29.3078 27.2727 -31.8182 C 27.2727 -34.3285 25.2377 -36.3636 22.7273 -36.3636 H 18.1818 C 15.6714 -36.3636 13.6364 -34.3268 13.6364 -31.8165 V -2.295 C 11.6114 -1.0009 9.2532 0 6.8182 0 C 4.6114 0 2.3659 -0.8223 -0.9341 -2.0814 C -3.7286 -3.1482 -7.3914 -4.5455 -11.3636 -4.5455 C -13.8893 -4.5455 -16.2083 -3.9977 -18.1818 -3.27 L -18.1818 -31.8182 C -18.1818 -34.294 -20.2515 -36.3636 -22.7273 -36.3636 H -27.2727 Z M 6.8182 9.0909 C 9.3441 9.0909 11.6627 8.5432 13.6364 7.8155 V 22.7273 C 13.6364 26.9636 12.3782 31.6436 9.7755 35.1445 C 7.2941 38.4827 3.5023 40.9091 -2.2727 40.9091 C -8.0477 40.9091 -11.8395 38.4827 -14.3209 35.1445 C -16.9234 31.6436 -18.1818 26.9636 -18.1818 22.7273 L -18.1818 6.8405 C -16.1567 5.5464 -13.7985 4.5455 -11.3636 4.5455 C -9.1568 4.5455 -6.9114 5.3677 -3.6114 6.6268 C -0.8168 7.6936 2.8459 9.0909 6.8182 9.0909 Z",
		cellDeath: "M 33.3333 45.2381 C 33.3333 47.8681 31.2014 50 28.5714 50 H 9.5238 H -9.5238 H -28.5714 C -31.2014 50 -33.3333 47.8681 -33.3333 45.2381 V 34.5238 C -33.3333 29.9214 -37.0643 26.1905 -41.6667 26.1905 C -44.9541 26.1905 -47.619 23.5257 -47.619 20.2381 V -2.381 C -47.619 -28.6802 -26.2993 -50 0 -50 C 26.299 -50 47.619 -28.6802 47.619 -2.381 V 20.2381 C 47.619 23.5257 44.9543 26.1905 41.6667 26.1905 C 37.0643 26.1905 33.3333 29.9214 33.3333 34.5238 V 45.2381 Z M 23.8095 40.4762 V 34.5238 C 23.8095 25.8848 29.9443 18.6786 38.0952 17.0238 V -2.381 C 38.0952 -23.4204 21.0395 -40.4762 0 -40.4762 C -21.0394 -40.4762 -38.0952 -23.4204 -38.0952 -2.381 V 17.0238 C -29.9444 18.6786 -23.8095 25.8848 -23.8095 34.5238 V 40.4762 H -14.2857 V 30.9524 C -14.2857 28.3224 -12.1538 26.1905 -9.5238 26.1905 C -6.8938 26.1905 -4.7619 28.3224 -4.7619 30.9524 V 40.4762 H 4.7619 V 30.9524 C 4.7619 28.3224 6.8938 26.1905 9.5238 26.1905 C 12.1538 26.1905 14.2857 28.3224 14.2857 30.9524 V 40.4762 H 23.8095 Z M -4.7619 4.7619 C -4.7619 11.3367 -16.0368 16.6667 -22.6072 16.6667 C -28.6443 16.6667 -28.6092 12.1667 -28.5638 6.3362 V 6.3357 C -28.5598 5.821 -28.5557 5.2957 -28.5557 4.7619 C -28.5557 -1.8129 -23.2292 -7.1429 -16.6588 -7.1429 C -10.0883 -7.1429 -4.7619 -1.8129 -4.7619 4.7619 Z M 28.5638 6.3362 C 28.5595 5.821 28.5557 5.2957 28.5557 4.7619 C 28.5557 -1.8129 23.229 -7.1429 16.6586 -7.1429 C 10.0881 -7.1429 4.7619 -1.8129 4.7619 4.7619 C 4.7619 11.3367 16.0367 16.6667 22.6071 16.6667 C 28.6443 16.6667 28.609 12.1667 28.5638 6.3362 Z",
	}
	// Color with which to draw icons and foreground text.
	fgColor := "#1e1e1e"
	if f.Owner == 2 {
		fgColor = "#7a6505"
	}
	// Coordinates for a hexagon with side length a, centered at (0, 0).
	a := sideLength
	b := math.Sqrt(3) * a
	ps := []float64{
		b / 2, a / 2, 0, a, -b / 2, a / 2, -b / 2, -a / 2, 0, -a, b / 2, -a / 2,
	}
	coords := make([]string, len(ps)/2)
	for i := range coords {
		coords[i] = fmt.Sprintf("%.6f,%.6f", ps[2*i], ps[2*i+1])
	}
	var fill string
	switch {
	case f.Owner == 1:
		fill = "#255ab4"
	case f.Owner == 2:
		fill = "#f8d748"
	case f.Type == cellGrass:
		fill = "#008048"
	case f.Type == cellRock:
		fill = "#5f5f5f"
	case f.Blocked > 0:
		fill = map[int]string{
			1: "#fbeba3",
			2: "#92acd9",
			3: "#5f5f5f",
		}[int(f.Blocked)]
	case es.Max() > 0:
		// #efbbff
		bgColor, err := ScaleRGB("#efbbff", "#800080", float64(es.Max()))
		if err != nil {
			fill = "none"
			break
		}
		fill = bgColor
		fgColor = "#303030"
	default:
		fill = "none"
	}
	stroke := "#cbcbcb"
	strokeWidth := 1
	if move != nil && r == move.Row && c == move.Col {
		stroke = "#be398d"
		strokeWidth = 4
	}
	points := strings.Join(coords, " ")
	xOff := 0.0
	if r%2 == 1 {
		xOff = b / 2
	}
	x := xOff + float64(c)*b + b/2
	y := float64(r)*a*3/2 + a
	transform := fmt.Sprintf("translate(%.6f, %.6f)", x, y)
	elems := []string{
		fmt.Sprintf(`<polygon points="%s" stroke="%s" stroke-width="%d" fill="%s" />`, points, stroke, strokeWidth, fill),
	}
	if p, ok := cellIconPaths[f.Type]; ok {
		// Add cell icon overlay.
		iconScale := b / 50 / 3.5 // Same funny magic number as in game.js.
		elems = append(elems, fmt.Sprintf(`<g transform="scale(%.6f)"><path d="%s" fill-rule="evenodd" fill="%s" /></g>`, iconScale, p, fgColor))
	}
	if f.Value > 0 {
		elems = append(elems, fmt.Sprintf(`<text style="text-anchor: middle; alignment-baseline: middle; font: %dpx sans-serif;" fill="%s" x="0" y="0">%d</text>`, int(a), fgColor, f.Value))
	}
	if es.Max() > 0 {
		elems = append(elems, fmt.Sprintf(`<text style="text-anchor: middle; alignment-baseline: middle; font: %dpx sans-serif;" fill="%s" x="0" y="0">
	   			<tspan x="0" dy="0">%.1f</tspan>
	   			<tspan x="0" dy="1.2em">%.1f</tspan>
	 			</text>`, int(a/2), fgColor, 100*es.flag, 100*es.normal))
	}
	return fmt.Sprintf(`<g transform="%s">%s</g>`, transform, strings.Join(elems, ""))
}

func moveScoreForKind(s *pb.SuggestMoveStats_ScoredMove, scoreKind pb.SuggestMoveStats_ScoreKind) (float32, bool) {
	for _, s := range s.Scores {
		if s.Kind == scoreKind {
			return s.Score, true
		}
	}
	return 0.0, false
}

func ExportSVG(file string, boards []*Board, captions []string) error {
	return ExportSVGWithStats(file, boards, nil, nil, pb.SuggestMoveStats_FINAL, captions)
}

// ExportSVG writes a HTML document to file that contains SVG renderings of the given boards.
// stats contains optional evaluation statistics (typically from MCTS),
// captions contains optional captions of the boards.
func ExportSVGWithStats(file string, boards []*Board, moves []*GameEngineMove, stats []*pb.SuggestMoveStats, scoreKind pb.SuggestMoveStats_ScoreKind, captions []string) error {
	const sideLength = 30.0
	width := 10 * math.Sqrt(3) * sideLength
	height := 17 * sideLength
	viewbox := fmt.Sprintf("0 0 %.6f %.6f", width, height)
	var sb strings.Builder
	sb.WriteString(`<!DOCTYPE html>
		<html>
		<head>
			<meta charset="utf-8" />
			<title>Hexz SVG Export</title>
			<style>
				body {
					background-color: #1e1e1e;
					color: #cbcbcb;
				}
				body {
					font-family: sans-serif;
				}
			</style>
		</head>
		<body>` + "\n")
	sb.WriteString("<h1>Hexz SVG Export</h1>\n")
	fmt.Fprintf(&sb, "<p>Created: %s</p>\n", time.Now().Format(time.RFC3339))
	for i, board := range boards {
		// Build map from cell indices to normal and flag evaluation scores.
		evalMap := make(map[idx]*cellEvalScores)
		if i < len(stats) && stats[i] != nil {
			for _, m := range stats[i].Moves {
				score, found := moveScoreForKind(m, scoreKind)
				if !found {
					continue // This cell has no move score information of the requested kind.
				}
				eval, ok := evalMap[idx{int(m.Row), int(m.Col)}]
				if !ok {
					eval = &cellEvalScores{}
				}
				switch m.Type {
				case pb.Field_NORMAL:
					eval.normal = score
				case pb.Field_FLAG:
					eval.flag = score
				}
				evalMap[idx{int(m.Row), int(m.Col)}] = eval
			}
		}
		fmt.Fprintf(&sb, "<h2>Board %d</h2>\n", i+1)
		// Add info line that is displayed above the board.
		infos := []string{
			fmt.Sprintf("Move: %d", board.Move),
			fmt.Sprintf("Turn: %d", board.Turn),
		}
		if len(board.Score) == 2 {
			infos = append(infos, fmt.Sprintf("Score: %d &ndash; %d", board.Score[0], board.Score[1]))
		}
		if i < len(stats) && stats[i] != nil {
			infos = append(infos, fmt.Sprintf("Value: %.3f", stats[i].Value))
		}
		fmt.Fprintf(&sb, "<p>"+strings.Join(infos, " &bullet; ")+"</p>\n")
		// Add SVG.
		var move *GameEngineMove
		if i < len(moves) && moves[i] != nil {
			move = moves[i]
		}
		sb.WriteString("<div>\n")
		fmt.Fprintf(&sb, `<svg xmlns="http://www.w3.org/2000/svg" width="%.6f" height="%.6f" viewBox="%s">`+"\n", width, height, viewbox)
		for r, row := range board.Fields {
			for c := range row {
				f := &board.Fields[r][c]
				hexpoly := hexPolySVG(sideLength, f, r, c, move, evalMap[idx{r, c}])
				sb.WriteString(hexpoly + "\n")
			}
		}
		sb.WriteString("</svg>\n")
		if i < len(captions) {
			fmt.Fprintf(&sb, "<p>%s</p>\n", captions[i])
		}
		sb.WriteString("</div>\n")
	}
	sb.WriteString("</body>\n")
	sb.WriteString("</html>\n")

	// Write output file.
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	defer f.Close()
	f.WriteString(sb.String())
	return nil
}
