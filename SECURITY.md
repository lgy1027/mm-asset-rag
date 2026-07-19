# Security Policy

## Reporting a Vulnerability

mm-asset-rag is an open-source project. If you discover a security
vulnerability, **please do not open a public GitHub issue**.

Report it privately by emailing the maintainer at the address listed in
the `pyproject.toml` `authors` field. Include:

- a description of the issue and its impact,
- the steps or a minimal reproduction,
- the version / commit you tested against, and
- any suggested fix if you have one.

We will acknowledge receipt within **5 business days** and aim to ship a
fix or mitigation within **30 days** for high-severity issues. Please do
not publicly disclose the vulnerability until a fix is released.

## Scope

This policy covers the `mm_asset_rag` package and the `mmrag` /
`mmrag-api` entry points. It does **not** cover:

- vulnerabilities in third-party dependencies (report those upstream), or
- issues that require already having access to a privileged host (the API
  binds `127.0.0.1` by default and is intended for loopback / trusted
  reverse-proxy deployment).

## Deployment hardening

Before exposing the API beyond localhost, review
[`docs/configuration.md`](docs/configuration.md) — in particular set
`MMRAG_API_TOKEN` (guards destructive + LLM-quota endpoints) and
`MMRAG_TRUSTED_HOSTS` (Host-header allow-list). The defaults are safe for
single-machine loopback use only.

> **Note on the bundled web UI:** the web UI served at `/` does **not** send
> an `Authorization` header, so once `MMRAG_API_TOKEN` is set its
> `/upload/*`, `/answer`, and `/chat/*` fetches return 401. The web UI is
> intended for the zero-config loopback default; on a token-guarded
> deployment use the HTTP API directly (or front the UI behind a proxy that
> injects the token). Read endpoints (`/search`, `/assets`, `/tasks`,
> `/health`) stay open regardless.
