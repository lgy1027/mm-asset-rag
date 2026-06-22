## Summary

<!-- One or two sentences: what does this PR change and why? -->

## Linked issues

<!-- Fixes #123, refs #456, etc. -->

## Changes

<!-- Bullet list of the user-facing / API-visible changes. -->

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (call out migration steps below)
- [ ] Docs only

## How was it tested?

<!-- Check what you actually ran. -->

- [ ] `pytest tests/unit -q`
- [ ] Manual: `mmrag-api` + browser smoke test (upload → chat → sources)
- [ ] Other: ____

## Migration / rollout notes

<!-- Breaking changes, env var additions, deployment steps. Delete this
section if there's nothing to call out. -->

## Checklist

- [ ] `ruff check . && ruff format .` is clean
- [ ] New env vars (if any) are documented in `.env.example`
- [ ] User-facing changes are mirrored in `README.md` / `docs/api.md`
- [ ] Commit messages follow `<type>(<scope>): <subject>`
- [ ] No AI co-author trailer (`Co-Authored-By: Claude ...`) or AI generation
      markers in any commit message, file header, or PR description