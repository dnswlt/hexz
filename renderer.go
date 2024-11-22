package hexz

import (
	"embed"
	"fmt"
	"html/template"
	"io"
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
)

func NewRenderer() (*Renderer, error) {
	tmpl, err := template.New("__root__").ParseFS(embeddedTemplates, "resources/templates/*.html")
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
