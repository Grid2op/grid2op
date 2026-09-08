# Release Process

This document describes the release methodology, criteria, and checklist for
this repository. It applies to all Core packages in the Grid2Op ecosystem
(grid2op, lightsim2grid, pypowsybl2grid).

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/) (`major.minor.patch`):

- **MAJOR** — incompatible API changes.
- **MINOR** — new backward-compatible functionality.
- **PATCH** — backward-compatible bug fixes.

---

## Release cadence

Releases are made on a rolling/ad hoc basis, as agreed upon by the committers
based on the readiness of features and fixes. There is no fixed cadence.

---

## Pre-release checklist

Before tagging a release, the release manager must verify the following:

### Code quality
- [ ] All CI checks pass on the target release branch (`dev_X.Y.Z`).
- [ ] No open issues tagged `release-blocker`.
- [ ] All deprecation warnings introduced in this cycle are documented.
- [ ] SCA scan passes with no unaddressed critical or high-severity findings (see [Software Composition Analysis](#software-composition-analysis-osps-vm-0502) below).

### Documentation
- [ ] `CHANGELOG.rst` has a complete entry for this version (features, fixes,
      breaking changes, deprecations).
- [ ] Public API documentation is up to date.
- [ ] Version number is updated in `pyproject.toml` (and `setup.py` /
      `CMakeLists.txt` if applicable).

### Tagging and branching
- [ ] A release PR merges `dev_X.Y.Z` into `master`.
- [ ] The release commit is tagged `vX.Y.Z` and the tag is signed.

---

## Building and publishing

### Pure-Python packages (grid2op)

```bash
python -m build          # produces sdist + wheel in dist/
twine check dist/*       # verify metadata before upload
twine upload dist/*      # upload to PyPI
```

### Compiled packages (lightsim2grid, pypowsybl2grid)

Binary wheels are built via the CI/CD pipeline (GitHub Actions + `cibuildwheel`)
for all supported platforms and Python versions before upload. Do not upload
manually built wheels.

---

## Software Bill of Materials (OSPS-QA-02.02)

> **Requirement:** When the project makes a release, all compiled released
> software assets **MUST** be delivered with a Software Bill of Materials
> (SBOM). [OSPS-QA-02.02]
>
> It is recommended to auto-generate SBOMs at build time using a tool that
> has been vetted for accuracy. This enables users to ingest this data in a
> standardized approach alongside other projects in their environment.

### What this means in practice

An SBOM is a machine-readable inventory of every component bundled in a
release artifact — direct and transitive dependencies, their versions, and
their licenses.

**grid2op** (pure Python): generating an SBOM is good practice and satisfies
the spirit of the requirement, even though the wheel contains no compiled code.

**lightsim2grid / pypowsybl2grid** (compiled wheels): the requirement applies
directly. Each release must ship an SBOM alongside the binary wheels, because
the compiled artifacts bundle vendored C++ libraries (Eigen, SuiteSparse/KLU,
CKTSO, etc.) that are otherwise invisible to downstream users.

### Recommended format

Use **CycloneDX JSON** (`application/vnd.cyclonedx+json`) or **SPDX**. Both
are widely supported by dependency-scanning tooling.

### How to generate automatically (GitHub Actions)

Add the following step to the release workflow, **after** the build step and
**before** the upload step:

```yaml
- name: Generate SBOM
  run: |
    pip install cyclonedx-bom
    # Captures the full Python environment (all installed deps)
    cyclonedx-py environment --output-format json --outfile sbom.cdx.json

- name: Attach SBOM to GitHub Release
  uses: softprops/action-gh-release@v2
  with:
    files: sbom.cdx.json
```

For **lightsim2grid**, the Python-level SBOM will not capture the vendored C++
libraries. Supplement it by either:

- Running `syft` on the built wheel file:
  ```yaml
  - uses: anchore/sbom-action@v0
    with:
      path: dist/
      format: cyclonedx-json
      output-file: sbom.cdx.json
  ```
- Or maintaining a hand-written `sbom-cpp-deps.cdx.json` that lists Eigen,
  SuiteSparse, KLU, and CKTSO with their pinned versions, and merging it into
  the final SBOM.

### Where to publish the SBOM

Attach `sbom.cdx.json` as a release asset on the GitHub Release page,
alongside the wheels and the source archive. Name it clearly, e.g.
`lightsim2grid-0.13.1-sbom.cdx.json`.

---

## Software Composition Analysis (OSPS-VM-05.02)

> **Requirement:** While active, the project documentation MUST include a
> policy to address SCA violations prior to any release. [OSPS-VM-05.02]

### Policy

No Core package may be released if any dependency has an unaddressed
critical or high-severity vulnerability (CVSS ≥ 7.0) known at the time of
release. Medium and low findings must be triaged and either remediated or
explicitly accepted with a written rationale before the release is tagged.

An exception may be granted by a TSC majority vote when no fix is available
upstream and the vulnerability is determined to be unexploitable in the
project's use context. Exceptions must be recorded in a `security-exceptions`
section of the repository's `SECURITY.md`, with the CVE identifier, the
rationale, and an agreed review date.

### Scope

**grid2op** (pure Python): the SCA scan covers all direct and transitive
Python dependencies as resolved in the release environment.

**lightsim2grid / pypowsybl2grid** (compiled wheels): the scan covers Python
dependencies **and** the vendored C++ libraries (Eigen, SuiteSparse/KLU,
CKTSO, etc.). Because `pip-audit` does not reach compiled vendored code, the
C++ dependency list must be reviewed manually against the
[OSV database](https://osv.dev/) or equivalent prior to each release.

### Recommended tool

Use `pip-audit` for Python dependencies:

```bash
pip install pip-audit
pip-audit --strict   # exits non-zero on any finding
```

`--strict` treats findings as errors. Use `--ignore-vuln <ID>` only for
vulnerabilities covered by an accepted exception in `SECURITY.md`.

### How to enforce automatically (GitHub Actions)

Add a required status check to the release workflow, **before** the build
step:

```yaml
- name: SCA — Python dependencies
  run: |
    pip install pip-audit
    pip-audit --strict
```

For **lightsim2grid**, add a manual review step as a checklist item (see
below) since tooling cannot automatically scan vendored C++ libraries.

Mark this job as a required status check in the branch protection rules for
`master` so that the release PR cannot be merged if it fails.

### Pre-release checklist additions

The following items are appended to the pre-release checklist above:

- [ ] `pip-audit --strict` passes on the release environment with no
      unaddressed findings.
- [ ] *(lightsim2grid / pypowsybl2grid only)* Vendored C++ libraries (Eigen,
      SuiteSparse/KLU, CKTSO) reviewed against OSV or NVD; findings recorded
      or accepted in `SECURITY.md`.
- [ ] Any accepted exceptions in `SECURITY.md` have been reviewed and are
      still valid for this release.

---

## Post-release checklist

- [ ] GitHub Release page created with the correct tag, release notes (copied
      from `CHANGELOG.rst`), and SBOM asset attached.
- [ ] PyPI page reflects the new version.
- [ ] Announcement posted to the Grid2Op mailing list and Discord.
- [ ] `dev_X.Y.(Z+1)` (or `dev_X.(Y+1).0`) branch opened for the next cycle.
- [ ] check the doc is updated in https://app.readthedocs.org and trigger 
      the build of the dev branch
- [ ] add the new dev branch to codacy (app.codacy.com)
