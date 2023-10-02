package hlog

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path"
	"runtime"
	"time"
)

var (
	L = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		AddSource: true,
		ReplaceAttr: func(groups []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				// We want to log the base name of the file, not the full path.
				if src, ok := attr.Value.Any().(*slog.Source); ok {
					src.File = path.Base(src.File)
				}
			}
			return attr
		},
	}))
)

func Infof(format string, args ...any) {
	if !L.Enabled(context.Background(), slog.LevelInfo) {
		return
	}
	var pcs [1]uintptr
	runtime.Callers(2, pcs[:]) // skip [Callers, Infof]
	r := slog.NewRecord(time.Now(), slog.LevelInfo, fmt.Sprintf(format, args...), pcs[0])
	_ = L.Handler().Handle(context.Background(), r)
}

func Errorf(format string, args ...any) {
	if !L.Enabled(context.Background(), slog.LevelInfo) {
		return
	}
	var pcs [1]uintptr
	runtime.Callers(2, pcs[:]) // skip [Callers, Errorf]
	r := slog.NewRecord(time.Now(), slog.LevelError, fmt.Sprintf(format, args...), pcs[0])
	_ = L.Handler().Handle(context.Background(), r)
}

func Fatalf(format string, args ...any) {
	var pcs [1]uintptr
	runtime.Callers(2, pcs[:]) // skip [Callers, Fatalf]
	r := slog.NewRecord(time.Now(), slog.LevelError, fmt.Sprintf(format, args...), pcs[0])
	_ = L.Handler().Handle(context.Background(), r)
	os.Exit(1)
}
