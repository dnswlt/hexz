package users

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/dnswlt/hexz/internal/hexzsql"
	"github.com/dnswlt/hexz/internal/hlog"
	"github.com/dnswlt/hexz/internal/mail"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

type CreateUserParams struct {
	PlayerName string
	Email      string
	Password   string
	// The URL (without the activation token query parameter) that users should call to confirm their activation.
	VerifyURL *url.URL
}

type Service interface {
	// CreateUser adds a new user to the store.
	// The default implementation sends out a verification email.
	CreateUser(ctx context.Context, ps CreateUserParams) error
	// VerifyUser takes a verification token (typically taken from a URL query
	// parameter of an activation link), checks whether the token is valid,
	// not expired and assigned to a user account awaiting verification.
	// If the checks are successful, the user is marked as verified and
	// can be used to log in.
	// Otherwise, an error isreturned.
	VerifyUser(ctx context.Context, token string) error
}

type ServiceDefault struct {
	store      hexzsql.UserStore
	mailClient *mail.Client
}

func NewService(store hexzsql.UserStore, mailClient *mail.Client) *ServiceDefault {
	return &ServiceDefault{
		store:      store,
		mailClient: mailClient,
	}
}

func (s *ServiceDefault) CreateUser(ctx context.Context, ps CreateUserParams) error {
	pwhash, err := bcrypt.GenerateFromPassword([]byte(ps.Password), bcrypt.DefaultCost)
	if err != nil {
		hlog.Errorf("Failed to create password hash: %v", err)
		return fmt.Errorf("failed to create password hash: %v", err)
	}
	verificationToken := uuid.New().String()

	// Send verification email first, so we don't add users to the DB that cannot verify
	// themselves.
	verifyLink := *ps.VerifyURL
	q := verifyLink.Query()
	q.Set("token", verificationToken)
	verifyLink.RawQuery = q.Encode()
	err = s.mailClient.SendAccountVerificationMail(ctx, ps.Email, ps.PlayerName, &verifyLink)
	if err != nil {
		return fmt.Errorf("user not added: failed to send verification email: %v", err)
	}

	// Add user to DB
	user := &hexzsql.User{
		Email:             ps.Email,
		PlayerName:        ps.PlayerName,
		PasswordHash:      string(pwhash),
		AccountStatus:     hexzsql.AccountStatusNew,
		VerificationToken: verificationToken,
		TokenExpiry:       time.Now().Add(24 * time.Hour),
	}
	err = s.store.AddUser(ctx, user)
	if errors.Is(err, hexzsql.ErrUserAlreadyExists) {
		return err
	}
	if err != nil {
		return fmt.Errorf("could not create user: %v", err)
	}

	return nil
}

func (s *ServiceDefault) VerifyUser(ctx context.Context, token string) error {
	return s.store.VerifyUser(ctx, token)
}
