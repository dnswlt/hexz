package users

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"testing"
	"time"

	"github.com/dnswlt/hexz/internal/hexzsql"
	"github.com/dnswlt/hexz/internal/mail"
	"github.com/google/uuid"
	"github.com/mrz1836/postmark"
)

type FakeUserStore struct {
	users map[string]*hexzsql.User
}

func NewFakeUserStore() *FakeUserStore {
	return &FakeUserStore{
		users: make(map[string]*hexzsql.User),
	}
}

func (s *FakeUserStore) FindUser(ctx context.Context, email string) (*hexzsql.User, error) {
	if u, ok := s.users[email]; ok {
		return u, nil
	}
	return nil, hexzsql.ErrUserNotFound
}

func (s *FakeUserStore) AddUser(ctx context.Context, user *hexzsql.User) error {
	if _, ok := s.users[user.Email]; ok {
		return hexzsql.ErrUserAlreadyExists
	}
	id := uuid.New().String()
	user.ID = id
	s.users[user.Email] = user
	return nil
}

func (s *FakeUserStore) DeleteUser(ctx context.Context, email string) error {
	delete(s.users, email)
	return nil
}

func (s *FakeUserStore) VerifyUser(ctx context.Context, verificationToken string) error {
	for _, u := range s.users {
		if u.VerificationToken == verificationToken {
			u.VerificationToken = ""
			u.TokenExpiry = time.Time{}
			u.AccountStatus = hexzsql.AccountStatusActive
			return nil
		}
	}
	return hexzsql.ErrInvalidToken
}

func TestCreateUser(t *testing.T) {
	store := NewFakeUserStore()
	mailClient := mail.NewClient(mail.DebugLoggingMailer{}, "test.notify@example.com")
	s := NewService(store, mailClient)
	verifyURL, err := url.Parse("http://localhost/verify")
	if err != nil {
		t.Fatal(err)
	}
	email := "test.user@example.com"
	err = s.CreateUser(context.Background(), CreateUserParams{
		PlayerName: "Test",
		Email:      email,
		Password:   "fooB4r",
		VerifyURL:  verifyURL,
	})
	if err != nil {
		t.Fatalf("CreateUser failed: %v", err)
	}
	user := store.users[email]
	if user == nil {
		t.Fatalf("User wasnt' stored in the store")
	}
	if user.VerificationToken == "" {
		t.Errorf("User has no verification token")
	}
	if len(user.PasswordHash) < 50 {
		// bcrypt hashes have ~60 characters
		t.Errorf("Password hash is too short: %q", user.PasswordHash)
	}
	if time.Since(user.TokenExpiry) > -1*time.Hour {
		t.Errorf("Token expires too early: %v", user.TokenExpiry)
	}
}

type FailingMailer struct{}

func (m FailingMailer) SendEmail(ctx context.Context, email postmark.Email) (postmark.EmailResponse, error) {
	return postmark.EmailResponse{}, fmt.Errorf("failed to send mail")
}

func TestCreateUserMailSendingFails(t *testing.T) {
	store := NewFakeUserStore()
	mailClient := mail.NewClient(FailingMailer{}, "test.notify@example.com")
	s := NewService(store, mailClient)
	verifyURL, err := url.Parse("http://localhost/verify")
	if err != nil {
		t.Fatal(err)
	}
	email := "test.user@example.com"
	err = s.CreateUser(context.Background(), CreateUserParams{
		PlayerName: "Test",
		Email:      email,
		Password:   "fooB4r",
		VerifyURL:  verifyURL,
	})
	if err == nil {
		t.Fatalf("CreateUser did not fail despite using failing mailer")
	}
	_, ok := store.users[email]
	if ok {
		t.Fatalf("User was added to the store despite verification email failure")
	}
}

func TestValidateLoginSuccess(t *testing.T) {
	store := NewFakeUserStore()
	mailClient := mail.NewClient(mail.DebugLoggingMailer{}, "test.notify@example.com")
	s := NewService(store, mailClient)
	verifyURL, err := url.Parse("http://localhost/verify")
	if err != nil {
		t.Fatal(err)
	}
	email := "test.user@example.com"
	password := "fooB4r"
	err = s.CreateUser(context.Background(), CreateUserParams{
		PlayerName: "Test",
		Email:      email,
		Password:   password,
		VerifyURL:  verifyURL,
	})
	if err != nil {
		t.Fatalf("CreateUser failed: %v", err)
	}
	user := store.users[email]
	if user == nil {
		t.Fatalf("User wasnt' stored in the store")
	}
	err = s.VerifyUser(context.Background(), user.VerificationToken)
	if err != nil {
		t.Fatalf("VerifyUser failed: %v", err)
	}
	vUser, err := s.ValidateLogin(context.Background(), email, password)
	if err != nil {
		t.Fatalf("ValidateLogin failed: %v", err)
	}
	if vUser.Email != user.Email {
		t.Errorf("Emails differ: want %q, got %q", user.Email, vUser.Email)
	}
}

func TestValidateLoginNotVerified(t *testing.T) {
	store := NewFakeUserStore()
	mailClient := mail.NewClient(mail.DebugLoggingMailer{}, "test.notify@example.com")
	s := NewService(store, mailClient)
	verifyURL, err := url.Parse("http://localhost/verify")
	if err != nil {
		t.Fatal(err)
	}
	email := "test.user@example.com"
	password := "fooB4r"
	err = s.CreateUser(context.Background(), CreateUserParams{
		PlayerName: "Test",
		Email:      email,
		Password:   password,
		VerifyURL:  verifyURL,
	})
	if err != nil {
		t.Fatalf("CreateUser failed: %v", err)
	}
	user := store.users[email]
	if user == nil {
		t.Fatalf("User wasnt' stored in the store")
	}
	_, err = s.ValidateLogin(context.Background(), email, password)
	if !errors.Is(err, ErrUserAccountNotActive) {
		t.Errorf("ValidateLogin did not fail for unverified user with expected error: %v", err)
	}
}
