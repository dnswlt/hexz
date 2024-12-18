package mail

import (
	"context"
	"net/url"
	"strings"
	"testing"

	"github.com/mrz1836/postmark"
)

type FakeClient struct {
	emails []postmark.Email
}

func (c *FakeClient) SendEmail(ctx context.Context, email postmark.Email) (postmark.EmailResponse, error) {
	c.emails = append(c.emails, email)
	return postmark.EmailResponse{}, nil
}

func TestPasswordResetMail(t *testing.T) {
	fakeClient := &FakeClient{}
	client := &Client{
		from:   "dummy@example.com",
		mailer: fakeClient,
	}
	resetLink, err := url.Parse("http://example.com/reset?token=abcdefgh")
	if err != nil {
		t.Fatal("invalid URL in test:", err)
	}
	err = client.SendPasswordResetMail(context.Background(), "foo@example.com", "foo", resetLink)
	if err != nil {
		t.Fatalf("send failed: %v", err)
	}
	if len(fakeClient.emails) != 1 {
		t.Fatalf("Expected one email to be sent, got %d", len(fakeClient.emails))
	}
	email := fakeClient.emails[0]
	if !strings.Contains(email.TextBody, resetLink.String()) {
		t.Errorf("Missing reset link in email: %s", email.TextBody)
	}
}
