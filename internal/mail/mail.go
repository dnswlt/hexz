package mail

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"text/template"

	_ "embed"

	"github.com/mrz1836/postmark"
)

//go:embed pw_reset.txt
var embeddedPasswordResetTmpl string

//go:embed account_verify.txt
var embeddedAccountVerificationTmpl string

// Mailer is the interface that clients needs to implement in order to fake out the real postmark client.
type Mailer interface {
	SendEmail(ctx context.Context, email postmark.Email) (postmark.EmailResponse, error)
}

type Client struct {
	from   string // The sender address to use in the From: header
	mailer Mailer
}

func NewClient(mailer Mailer, from string) *Client {
	return &Client{
		from:   from,
		mailer: mailer,
	}
}

func NewPostmarkClient(serverToken string, from string) *Client {
	// Account token should not be required for our purposes.
	accountToken := ""
	return &Client{
		from:   from,
		mailer: postmark.NewClient(serverToken, accountToken),
	}
}

func (c *Client) SendPasswordResetMail(ctx context.Context, to string, username string, resetLink *url.URL) error {
	tmpl, err := template.New("password").Parse(embeddedPasswordResetTmpl)
	if err != nil {
		return fmt.Errorf("cannot parse password reset template: %v", err)
	}
	var textBody strings.Builder
	tmpl.Execute(&textBody, map[string]any{
		"Username":  username,
		"ResetLink": resetLink.String(),
	})
	email := postmark.Email{
		From:       c.from,
		To:         to,
		Subject:    "Your Hexz Password Reset Link",
		TextBody:   textBody.String(),
		Tag:        "pw-reset",
		TrackOpens: false,
	}
	resp, err := c.mailer.SendEmail(ctx, email)
	if err != nil {
		return fmt.Errorf("failed to send password reset email: %v", err)
	}
	if resp.ErrorCode != 0 {
		return fmt.Errorf("postmark returned a non-zero error: %d: %s", resp.ErrorCode, resp.Message)
	}
	return nil
}

func (c *Client) SendAccountVerificationMail(ctx context.Context, to string, username string, verifyLink *url.URL) error {
	tmpl, err := template.New("account").Parse(embeddedAccountVerificationTmpl)
	if err != nil {
		return fmt.Errorf("cannot parse account verification template: %v", err)
	}
	var textBody strings.Builder
	tmpl.Execute(&textBody, map[string]any{
		"Username":         username,
		"VerificationLink": verifyLink.String(),
	})
	email := postmark.Email{
		From:       c.from,
		To:         to,
		Subject:    "Verify Your Hexz Account",
		TextBody:   textBody.String(),
		Tag:        "acct-verify",
		TrackOpens: false,
	}
	resp, err := c.mailer.SendEmail(ctx, email)
	if err != nil {
		return fmt.Errorf("failed to send account verification email: %v", err)
	}
	if resp.ErrorCode != 0 {
		return fmt.Errorf("postmark returned a non-zero error: %d: %s", resp.ErrorCode, resp.Message)
	}
	return nil
}
