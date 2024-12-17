package main

import (
	"context"
	"log"
	"net/url"
	"os"
	"path"

	"github.com/dnswlt/hexz/internal/mail"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatalf("Usage: %s <from-address> <to-address>", path.Base(os.Args[0]))
	}

	serverToken := os.Getenv("HEXZ_POSTMARK_SERVER_TOKEN")
	if serverToken == "" {
		log.Fatal("HEXZ_POSTMARK_SERVER_TOKEN is not set")
	}

	fromAddress := os.Args[1]
	toAddress := os.Args[2]

	client := mail.NewPostmarkClient(serverToken, fromAddress)
	resetLink, err := url.Parse("https://example.com/resetpw?token=123123123")
	if err != nil {
		log.Fatal("invalid url:", err)
	}
	err = client.SendPasswordResetMail(context.Background(), toAddress, "superuser", resetLink)
	if err != nil {
		log.Fatalf("Failed to send mail: %v", err)
	}
}
