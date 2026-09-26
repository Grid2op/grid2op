# Secrets and Credentials Management Policy

**Project:** Grid2Op (LF Energy Foundation)  
**Requirement:** OSPS-BR-07.02  
**Status:** Active  

---

## 1. Overview

This document defines how the Grid2Op project manages secrets and credentials — including API keys, tokens, passwords, certificates, and any other sensitive information. All contributors, committers, and Technical Steering Committee (TSC) members are expected to follow this policy.

The core principle is simple: **no secret should ever be committed to version control.**

---

## 2. Scope

This policy applies to:

- All repositories under the Grid2Op GitHub organization
- CI/CD pipelines (GitHub Actions workflows)
- Maintainer and contributor tooling
- Any third-party service integrations (PyPI publishing, cloud testing environments, etc.)

It covers secrets such as:

- API keys (PyPI tokens, cloud provider keys, etc.)
- Authentication tokens (GitHub tokens, OAuth tokens)
- Passwords and passphrases
- Private keys and certificates
- Webhook secrets

---

## 3. Storage

### 3.1 What is prohibited

The following are strictly forbidden across all repositories:

- Hard-coding secrets in source code (`.py`, `.cpp`, `.cu`, `.h`, scripts, etc.)
- Storing secrets in configuration files checked into version control (`.env`, `config.yaml`, `settings.json`, etc.)
- Including secrets in documentation, comments, or test fixtures
- Logging secrets to standard output or error streams

### 3.2 Approved storage locations

| Secret type | Approved storage |
|---|---|
| CI/CD secrets (tokens, keys used in workflows) | GitHub Actions Encrypted Secrets (repository or organization level) |
| Local development credentials | Developer's local `.env` file (must be listed in `.gitignore`) |
| Shared maintainer credentials | GitHub Actions organization-level secrets, accessible only to designated workflows |

All repositories **must** include a `.gitignore` that excludes `.env`, `.env.*`, `*.key`, `*.pem`, and similar file patterns.

---

## 4. Access Control

- **Principle of least privilege:** each secret is accessible only to the workflow or person that needs it, and nothing more.
- GitHub Actions secrets are scoped at the repository level by default. Organization-level secrets are granted only when multiple repositories legitimately need the same credential (e.g., a shared PyPI publishing token).
- Only TSC members and designated committers may add or modify secrets in the GitHub organization settings.
- Access to secrets is reviewed at each annual project review and whenever a committer or TSC member steps down (see Section 5).

---

## 5. Rotation and Revocation

### 5.1 Scheduled rotation

- **PyPI and package publishing tokens:** rotated at least once per year, or upon a new release cycle if that is shorter.
- **Long-lived API keys:** rotated at least once per year.

### 5.2 Event-triggered rotation

Secrets **must** be rotated immediately when:

- A committer or TSC member with access to secrets leaves the project or moves to emeritus status.
- A secret is accidentally exposed (committed to a repo, logged, shared in a public channel, etc.).
- A third-party service reports a breach or compromise.
- A repository is forked publicly under circumstances that may have exposed workflow secrets.

### 5.3 Incident response

If a secret is suspected or confirmed to have been exposed:

1. **Revoke** the secret immediately via the relevant platform (GitHub, PyPI, etc.).
2. **Notify** the TSC and the LF Energy security contact within 24 hours.
3. **Audit** recent usage of the secret for unauthorized activity.
4. **Replace** with a new secret following the storage and access guidelines above.
5. **Document** the incident in a private TSC record.

---

## 6. CI/CD Pipelines

- Secrets are injected into GitHub Actions workflows exclusively via `${{ secrets.SECRET_NAME }}` syntax.
- Workflows must never print secret values, even for debugging. Use `::add-mask::` if a dynamic value derived from a secret needs to appear in logs.
- Pull requests from forks do **not** have access to repository secrets by default; this GitHub Actions default must not be overridden.
- Workflow files (`.github/workflows/*.yml`) are reviewed by at least one committer before merge, with attention to any `env:` blocks that could inadvertently expose secrets.

---

## 7. Tiered Repository Considerations

Consistent with the Grid2Op "Core" and "Examples" project structure:

- **Core repositories** (grid2op, lightsim2grid, pypowsybl2grid): this full policy applies without exception.
- **Example/companion repositories**: the prohibition on hard-coded secrets and the incident response procedure apply unconditionally. Rotation schedules may be relaxed if a repository has no active CI publishing credentials, but this must be explicitly documented in that repository's README.
- **Archived repositories**: any active secrets must be revoked before archiving, and the archive notice should confirm that no live credentials are present.

---

## 8. Developer Guidance

Contributors are encouraged to:

- Use [`detect-secrets`](https://github.com/Yelp/detect-secrets) as a local pre-commit hook to catch accidental commits (see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for setup instructions).
- Store local development credentials in a `.env` file at the project root (never committed).
- Use environment variable loading libraries (e.g., `python-dotenv`) rather than hard-coding values.

Example `.env` usage in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()  # loads from .env file if present
api_key = os.environ["MY_API_KEY"]  # raises KeyError if not set — intentional
```

---

## 9. Compliance and Enforcement

- Compliance with this policy is a condition of maintaining committer or TSC status.
- Automated scanning (e.g., GitHub secret scanning, which is enabled on all public repositories by default) is used to detect accidental exposure.
- Any pull request found to contain secrets will be closed immediately. The contributor will be notified and asked to rotate the exposed credential before resubmitting.

---

## 10. Policy Maintenance

This document is owned by the TSC. It must be reviewed:

- At each annual LF Energy project review
- Whenever a significant change to project infrastructure occurs
- Following any security incident

Proposed changes follow the standard pull request process, with approval from a majority of active TSC members.

---

*For questions or to report a suspected exposure, contact the TSC via the project's [mailing list](https://lists.lfenergy.org/g/grid2op-discussion) or the [Discord server](https://discord.gg/cYsYrPT).*