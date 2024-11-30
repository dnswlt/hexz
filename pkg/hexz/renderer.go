package hexz

import (
	"fmt"
	"html/template"
	"io"
	"path"
	"time"

	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

type Renderer struct {
	templateDir string
	tmpl        *template.Template
	autoReload  bool
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
		"shortdate": func(t time.Time) string {
			return t.Local().Format("02/01 15:04")
		},
		"cpuPlayerMode": func(e pb.CPUPlayerMode_Enum) string {
			switch e {
			case pb.CPUPlayerMode_NONE:
				return "2P"
			case pb.CPUPlayerMode_EMBEDDED_CPU:
				return "1P e"
			case pb.CPUPlayerMode_REMOTE_CPU:
				return "1P r"
			case pb.CPUPlayerMode_WASM:
				return "1P w"
			default:
				return "?"
			}
		},
	}
}

func newTemplate(templateDir string) (*template.Template, error) {
	return template.New("__root__").Funcs(commonFuncs()).ParseGlob(path.Join(templateDir, "*.html"))
}

// NewRenderer creates a new Renderer that reads templates from the given templates folder.
// That folder is expected to contain the *.html template files (no subdirs).
func NewRenderer(templateDir string) (*Renderer, error) {
	tmpl, err := newTemplate(templateDir)
	if err != nil {
		return nil, fmt.Errorf("cannot create templates: %v", err)
	}
	return &Renderer{
		templateDir: templateDir,
		tmpl:        tmpl,
		autoReload:  false,
	}, nil
}

func (r *Renderer) SetAutoReload(enabled bool) {
	r.autoReload = enabled
}

func (r *Renderer) Render(w io.Writer, filename string, data map[string]any) error {
	tmpl := r.tmpl
	if r.autoReload {
		// Always read templates from disk in debug mode.
		var err error
		tmpl, err = newTemplate(r.templateDir)
		if err != nil {
			return fmt.Errorf("cannot create templates: %v", err)
		}

	}
	return tmpl.ExecuteTemplate(w, filename, data)
}
