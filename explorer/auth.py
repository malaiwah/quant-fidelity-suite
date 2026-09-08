"""Per-caller HF identity. No ambient/server-token fallback and no token serialization."""
from __future__ import annotations

import os
import time
from urllib.parse import urlsplit


class AuthError(ValueError):
    pass


class Actor:
    __slots__ = ("username", "expires_at", "source", "_token")

    def __init__(self, username, token, *, expires_at=None, source="bearer"):
        self.username = username
        self.expires_at = expires_at
        self.source = source
        self._token = token

    def __repr__(self):
        return "Actor(username=%r, token=<redacted>)" % self.username

    def client(self):
        from huggingface_hub import HfApi
        return HfApi(endpoint="https://huggingface.co", token=self._token)

    def public(self):
        return {"username": self.username, "expires_at": self.expires_at, "authentication": self.source}

    def require_lifetime(self, seconds):
        if self.expires_at is not None and self.expires_at <= time.time() + seconds:
            raise AuthError("Your sign-in expires before this operation can finish. Sign in again.")


def _valid_token(value):
    return isinstance(value, str) and 16 <= len(value) <= 4096 and not any(c.isspace() for c in value)


def actor_from_token(token, *, expires_at=None, source="bearer", profile_name=None):
    """Explicit-token API boundary; never reads HF_TOKEN or the local auth cache."""
    if not _valid_token(token):
        raise AuthError("Sign in with Hugging Face before accessing Jobs or publication.")
    if expires_at is not None and expires_at <= time.time():
        raise AuthError("Your Hugging Face sign-in has expired. Sign in again.")
    from huggingface_hub import HfApi
    try:
        identity = HfApi(endpoint="https://huggingface.co", token=token).whoami()
    except Exception:
        raise AuthError("Hugging Face could not validate this caller's credentials. Sign in again or check token permissions.") from None
    username = identity.get("name")
    if not isinstance(username, str) or not username or "/" in username:
        raise AuthError("The authenticated account has no valid personal namespace.")
    if profile_name is not None and username != profile_name:
        raise AuthError("The sign-in profile and access-token identity disagree.")
    return Actor(username, token, expires_at=expires_at, source=source)


def actor_from_request(request, oauth_profile=None, oauth_token=None):
    """Accept injected real HF OAuth or an explicit API Bearer token only.

    Cookie-authenticated browser mutations require the Space's own origin. A
    Bearer API request proves possession separately and never uses OAuth cookies.
    Gradio's convenient local mocked OAuth must never acquire spending authority.
    """
    headers = getattr(request, "headers", {}) or {}
    authorization = headers.get("authorization") or headers.get("Authorization")
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() != "bearer":
            raise AuthError("API authentication requires an explicit Bearer token.")
        return actor_from_token(value, source="bearer")
    if oauth_profile is None or oauth_token is None:
        raise AuthError("Sign in with Hugging Face to use your own Jobs and private results.")
    if not all(os.environ.get(k) for k in ("SPACE_ID", "SPACE_HOST", "OAUTH_CLIENT_ID", "OAUTH_CLIENT_SECRET")):
        raise AuthError("Local/mock OAuth cannot authorize Jobs or publication. Use an explicit API token for local development.")
    origin = headers.get("origin") or headers.get("Origin")
    host = os.environ["SPACE_HOST"].removeprefix("https://").rstrip("/")
    allowed = "https://" + host
    if not isinstance(origin, str) or origin.rstrip("/") != allowed:
        raise AuthError("Browser actions must originate in this Space. Open its direct URL and sign in again.")
    parsed = urlsplit(allowed)
    if not parsed.hostname or not parsed.hostname.endswith(".hf.space") or parsed.path:
        raise AuthError("The Space origin is not valid for privileged browser actions.")
    profile_name = getattr(oauth_profile, "username", None) or oauth_profile.get("preferred_username")
    return actor_from_token(oauth_token.token, expires_at=oauth_token.expires_at,
                            source="oauth", profile_name=profile_name)


def require_registry_owner(actor, registry_repository):
    namespace = registry_repository.split("/", 1)[0]
    if actor.username != namespace:
        raise AuthError("Registry acceptance is restricted to its authenticated personal namespace owner.")
    return actor
