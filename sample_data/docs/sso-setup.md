# Single Sign-On Setup

## Google Workspace SSO (Growth and above)

Go to Settings > Security > Single sign-on, choose Google Workspace and enter your domain.
Members who sign in with a Google account on that domain join the workspace automatically
with the Viewer role.

## SAML SSO (Scale and Enterprise)

Lumora supports SAML 2.0 with Okta, Microsoft Entra ID and OneLogin. Upload your identity
provider's metadata XML in Settings > Security > SAML. The Assertion Consumer Service URL is
https://auth.lumora.example/saml/acs.

After enabling SAML you can turn on "Enforce SSO", which blocks password logins for everyone
except workspace owners. Keep at least one owner with a password as a break-glass account.

## SCIM provisioning

SCIM 2.0 provisioning is available on Enterprise only. It creates, updates and deactivates
members automatically when they change in your identity provider.
