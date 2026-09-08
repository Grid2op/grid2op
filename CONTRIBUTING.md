# Contributing to Grid2Op

Grid2Op is an [LF Energy](https://lfenergy.org/) project, and we welcome
contributions from everyone — power systems engineers, machine learning
researchers, software developers, students, and curious newcomers.

This document explains how to get involved, what the contribution workflow
looks like, and what we expect from contributions to the codebase.

If anything here is unclear or out of date, please open an issue or a pull
request — improving this document is itself a valuable contribution.

---

## Table of contents

- [Code of Conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Where to start](#where-to-start)
- [Reporting bugs and security issues](#reporting-bugs-and-security-issues)
- [Code contribution workflow](#code-contribution-workflow)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Pre-commit hooks](#pre-commit-hooks)
- [Code style](#code-style)
- [Tests](#tests)
- [Documentation](#documentation)
- [Commit messages and pull requests](#commit-messages-and-pull-requests)
- [Becoming a committer or maintainer](#becoming-a-committer-or-maintainer)
- [License of contributions](#license-of-contributions)
- [Getting help](#getting-help)

---

## Code of Conduct

All participants in the Grid2Op community are expected to follow the project's
[Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to uphold
a friendly, welcoming, and harassment-free environment.

Reports of unacceptable behavior can be sent to the project maintainers via
the contacts listed in the Code of Conduct.

---

## Ways to contribute

Contributions don't have to be code. All of the following are valuable:

- Reporting bugs and reproducing issues filed by others.
- Improving the documentation, including docstrings and the
  [official docs](https://grid2op.readthedocs.io/).
- Improving or adding example notebooks in `getting_started/` and `examples/`.
- Fixing issues — especially those tagged
  [`good first issue`](https://github.com/Grid2op/grid2op/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  or [`help wanted`](https://github.com/Grid2op/grid2op/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
- Adding new functionality, optimizations, or new backends or type of observation.
- Reviewing pull requests opened by other contributors.
- Helping users on the project [Discord](https://discord.gg/cYsYrPT) and
  [GitHub Discussions](https://github.com/orgs/Grid2op/discussions).
- Writing blog posts, tutorials, or talks about Grid2Op.

---

## Where to start

If you want to contribute code but aren't sure where to begin:

- Browse issues labeled
  [`good first issue`](https://github.com/Grid2op/grid2op/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  and [`help wanted`](https://github.com/Grid2op/grid2op/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
- Check the **Work "in progress"** section at the top of
  [`CHANGELOG.rst`](./CHANGELOG.rst) for ideas the maintainers have flagged.
- Ask in [Discussions](https://github.com/orgs/Grid2op/discussions) or on
  [Discord](https://discord.gg/cYsYrPT) — we're happy to help you find
  something that matches your interests and skill level.

For larger contributions, **please open an issue first** to discuss the design
before writing code. This avoids duplicate work and ensures the change fits
the project's direction.

---

## Reporting bugs and security issues

### Regular bugs

Open a [GitHub issue](https://github.com/Grid2op/grid2op/issues) and include:

- Grid2Op version (`pip show grid2op`).
- Python version and operating system.
- Backend in use (PandaPower, lightsim2grid, pypowsybl2grid, custom).
- A minimal, reproducible code example.
- The full traceback or unexpected behavior, with what you expected instead.

### Security vulnerabilities

**Do not open a public issue for security vulnerabilities.** Follow the
disclosure process described in [`SECURITY.md`](./SECURITY.md). The
maintainers will coordinate a fix and a coordinated disclosure.

---

## Code contribution workflow

1. **Fork** the repository at
   [`https://github.com/Grid2op/grid2op`](https://github.com/Grid2op/grid2op).

2. **Identify the active development branch.** Grid2Op uses a `dev_X.Y.Z`
   convention. The active development branch is the one with the highest
   `dev_*` number on the official repository — it targets the next release.
   For example, if the latest release on PyPI is `1.12.2`, the active
   development branch will typically be `dev_1.12.3` or `dev_1.13.0`.

3. **Create a topic branch** off that development branch in your fork
   (e.g. `fix/observation-typing` or `feature/new-reward`). Do not work
   directly on `master` or on `dev_*` branches in your fork.

4. **Set up a development install:**

   ```bash
   git clone https://github.com/<your-username>/grid2op.git
   cd grid2op
   git remote add upstream https://github.com/Grid2op/grid2op.git
   pip install -e .[optional,docs]
   ```

5. **Implement your changes**, including:
   - Tests for new functionality or bug fixes (see [Tests](#tests)).
   - Updated documentation if your change affects user-facing behavior
     (see [Documentation](#documentation)).
   - An entry in `CHANGELOG.rst` under the appropriate section.

6. **Sign your commits** with the Developer Certificate of Origin
   (see [DCO](#developer-certificate-of-origin-dco)).

7. **Sync with the latest development branch** before opening the PR and
   resolve any conflicts:

   ```bash
   git fetch upstream
   git rebase upstream/dev_X.Y.Z
   ```

8. **Open a pull request** targeting the active `dev_X.Y.Z` branch (not
   `master`). Describe what you changed, why, and link any relevant issues.

---

## Developer Certificate of Origin (DCO)

Grid2Op uses the [Developer Certificate of Origin](https://developercertificate.org/)
to certify that contributors have the right to submit the code they contribute.
This is the same lightweight mechanism used by the Linux kernel and most
other Linux Foundation projects.

**Every commit must be signed off.** Add a `Signed-off-by` line to your
commit message by using the `-s` flag with `git commit`:

```bash
git commit -s -m "Fix typo in observation docstring"
```

This produces a trailer in the commit message like:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

The name and email must match the values configured in
`git config user.name` and `git config user.email`. By signing off, you
certify the statements in the [DCO text](https://developercertificate.org/).

A DCO check runs automatically on every pull request. PRs with unsigned
commits cannot be merged. If you forget to sign off, you can amend the
last commit with `git commit --amend -s` or rewrite history with
`git rebase --signoff <base>` for a series of commits.

---

## Pre-commit hooks

Grid2Op uses [pre-commit](https://pre-commit.com/) to run security and style checks
automatically before each commit. The hooks cover two stages:

- **pre-commit stage** — [`detect-secrets`](https://github.com/Yelp/detect-secrets)
  prevents accidentally committing API keys, tokens, or passwords, as required by the
  project's [Secrets Management Policy](./SECRETS_MANAGEMENT.md);
  [`check-mailmap`](https://github.com/jumanjihouse/pre-commit-hooks) checks author
  identity against `.mailmap`.
- **commit-msg stage** — a Python script (`scripts/check_dco_msg.py`) verifies that
  every commit message contains a `Signed-off-by` trailer (see
  [DCO](#developer-certificate-of-origin-dco)). Python is the only runtime
  dependency for this check — no extra tools are needed beyond what you already have.

Setting up the hooks is a one-time step and takes about two minutes.

### Installation

Install the required tools into your development virtual environment (the same one
you use for Grid2Op itself):

```bash
pip install pre-commit detect-secrets
```

### First-time setup

Run these two commands once after cloning (or after the hooks were first added to the
repo). Both are required: `pre-commit install` covers the file-level hooks, while the
`--hook-type commit-msg` flag activates the DCO check, which runs at the commit-msg
stage.

```bash
pre-commit install                          # pre-commit and mailmap hooks
pre-commit install --hook-type commit-msg   # DCO Signed-off-by check
```

### Normal workflow

Once installed, the hooks run automatically on every `git commit`. You will see
output like:

```
detect-secrets...............................................Passed
```

If a hook fails, the commit is blocked and the problem is described. Fix it, then
`git add` and retry.

To run all hooks manually without committing (useful before opening a PR):

```bash
pre-commit run --all-files
```

### Updating the baseline after a false positive

If a new file triggers a `detect-secrets` false positive (e.g. a test fixture
containing a dummy token string), update the baseline:

```bash
detect-secrets scan --baseline .secrets.baseline
detect-secrets audit .secrets.baseline   # mark the new finding as a false positive
git add .secrets.baseline
git commit -s -m "chore: update secrets baseline"
```

### Skipping hooks in exceptional cases

You can bypass hooks with `git commit --no-verify`, but **note that
`detect-secrets` also runs in CI** — a skipped local check will still be caught
on the pull request. Only skip if you have a specific, time-sensitive reason,
and follow up with a proper fix.

---

## Code style

We aim for code that is readable, well-typed where practical, and
consistent with the rest of the codebase.

- **Formatting:** follow the style of existing code in the file you are
  modifying. PEP 8 conventions apply by default.
- **Type hints:** encouraged for new code, especially in public APIs.
- **Docstrings:** use the Sphinx / reST format already used throughout the
  codebase. Public classes, functions, and methods must be documented.
- **Imports:** group standard library, third-party, then Grid2Op imports.
- **No commented-out code** in submitted PRs. Use git history if you need
  to recover something later.

The C++ code in dependent backends (e.g. `lightsim2grid`) follows the
conventions documented in those repositories.

---

## Tests

Grid2Op uses Python's standard `unittest` framework. The full suite lives
under `grid2op/tests/`.

To run the tests locally:

```bash
pip install -e .[optional]
cd grid2op/tests
python3 -m unittest discover
```

For new contributions:

- **Bug fixes** must include a test that fails before the fix and passes
  after.
- **New features** must include tests covering the new behavior, including
  edge cases.
- Tests should be deterministic. Seed any random number generators.
- Avoid tests that require network access or external services.

The full continuous integration suite runs on Linux with the latest NumPy
on Python 3.12. Smaller subsets run on Windows and macOS, and across
Python 3.8 through 3.13. Please make sure your changes do not regress
support for the supported Python versions documented in the README.

---

## Documentation

User-facing documentation lives in `docs/` and is built with Sphinx.

To build the docs locally:

```bash
pip install -e .[docs]
make html       # on Linux/macOS
# or:
sphinx-build -b html docs documentation   # any platform
```

The rendered output is placed in `documentation/html/index.html`.

When contributing:

- New public APIs must be added to the relevant `.rst` file in `docs/`.
- Code examples in docs and notebooks should be runnable as written.
- For Jupyter notebooks, **clear all outputs** before committing
  (`jupyter nbconvert --clear-output --inplace your_notebook.ipynb`).

---

## Commit messages and pull requests

- Use clear, present-tense commit subjects (e.g. `Fix race in Runner.run`,
  not `Fixed a race`).
- Keep the subject under ~72 characters; expand in the body if needed.
- Reference issues with `Fixes #123` or `Refs #123` in the commit body or
  PR description.
- Keep PRs focused. A PR that fixes a typo, adds a feature, and refactors
  three modules is harder to review than three separate PRs.
- Mark PRs as **Draft** if they are not ready for review.

---

## Becoming a committer or maintainer

Grid2Op is governed by a Technical Steering Committee (TSC) as documented
in [`GOVERNANCE.md`](./GOVERNANCE.md). Current committers and emeritus
members are listed in [`COMMITTERS.csv`](./COMMITTERS.csv).

Contributors who consistently provide high-quality contributions —
whether code, reviews, documentation, or community support — may be
invited to become committers by a vote of the TSC. There is no fixed
number of contributions required; we look for sustained engagement,
good technical judgment, and alignment with the project's values.

If you are interested in deeper involvement, the best path is to start
contributing regularly and reviewing other people's pull requests.

---

## License of contributions

Grid2Op is licensed under the
[Mozilla Public License 2.0 (MPL-2.0)](./LICENSE.md).

By submitting a contribution, you agree that your contribution will be
licensed under MPL-2.0 (inbound = outbound). Combined with the DCO
sign-off, this is the only licensing step required for contributions.

If your contribution includes code from another source, make sure that
source is compatible with MPL-2.0 and that the third-party license is
properly attributed in `LicensesInformation.md`.

---

## Getting help

- **Questions and discussion:**
  [GitHub Discussions](https://github.com/orgs/Grid2op/discussions)
- **Real-time chat:** [Grid2Op Discord](https://discord.gg/cYsYrPT)
- **Bug reports and feature requests:**
  [GitHub Issues](https://github.com/Grid2op/grid2op/issues)
- **Direct contact:** see the corresponding author on the
  [PyPI page](https://pypi.org/project/Grid2Op/).

Thanks for taking the time to contribute to Grid2Op!