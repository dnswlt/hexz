package hexz

import (
	"embed"
	"fmt"
	"html/template"
	"io"

	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

//go:embed resources/templates
var embeddedTemplates embed.FS

type Renderer struct {
	tmpl *template.Template
}

const (
	gameHtmlFilename    = "game.html"
	viewHtmlFilename    = "view.html"
	loginHtmlFilename   = "login.html"
	newGameHtmlFilename = "new.html"
	rulesHtmlFilename   = "rules.html"
	historyHtmlFilename = "history.html"
)

func commonFuncs() template.FuncMap {
	return map[string]any{
		"protodate": func(t *tpb.Timestamp) string {
			return t.AsTime().Local().Format("2006-01-02 15:04:05")
		},
		"cpuPlayerMode": func(e pb.CPUPlayerMode_Enum) string {
			switch e {
			case pb.CPUPlayerMode_NONE:
				return "2P"
			case pb.CPUPlayerMode_EMBEDDED_CPU:
				return "1P (embedded CPU)"
			case pb.CPUPlayerMode_REMOTE_CPU:
				return "1P (remote CPU)"
			case pb.CPUPlayerMode_WASM:
				return "1P (WASM)"
			default:
				return "?"
			}
		},
	}
}

func NewRenderer() (*Renderer, error) {
	tmpl, err := template.New("__root__").Funcs(commonFuncs()).ParseFS(embeddedTemplates, "resources/templates/*.html")
	if err != nil {
		return nil, fmt.Errorf("cannot create templates: %v", err)
	}
	return &Renderer{
		tmpl: tmpl,
	}, nil
}

func (r *Renderer) Render(w io.Writer, filename string, data map[string]any) error {
	return r.tmpl.ExecuteTemplate(w, filename, data)
}
