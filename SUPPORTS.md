# Support

Grid2Op is an [LF Energy](https://lfenergy.org/) project.
This document describes the support lifecycle for Grid2Op releases, including
the scope and duration of support, the types of support provided, and how to
obtain help.

---

## Release support lifecycle

Grid2Op uses a three-part version number `MAJOR.MINOR.PATCH` (e.g. `1.10.4`).
Security fixes are backported according to severity as follows:

| Severity | Releases patched |
|---|---|
| **Critical** | All patch releases of the last minor release published within the previous calendar year, **and** all patch releases of the current minor release train |
| **Important** | All patch releases of the last minor release published within the previous calendar year, **and** all patch releases of the current minor release train |
| **Moderate** | Latest patch release of the current minor release train only |
| **Low** | Included in the next scheduled release of the current release train |
| **Any severity — end of life** | Minor releases older than the last minor release published within the previous calendar year receive **no security updates of any kind** and should be considered end-of-life. Users are strongly encouraged to upgrade. |

**Example.** Suppose the current release is `1.11.1` and `1.10` was the last
minor release published within the previous calendar year:

- A *critical* vulnerability found in `1.10.4` would be patched in `1.10.0`,
  `1.10.1`, `1.10.2`, `1.10.3`, `1.10.4`, `1.11.0`, and `1.11.1`.
- A *moderate* vulnerability would only be patched in `1.11.1` (current latest
  patch).
- Any severity is also included in the next release of the active development
  branch.

Severity is assessed using the
[Apache severity rating](https://security.apache.org/blog/severityrating/).
When in doubt, report privately — see [`SECURITY.md`](./SECURITY.md).

> The list of currently supported versions is kept up to date in
> [`SECURITY.md`](./SECURITY.md).

---

## Types of support provided

For **supported releases**, the project aims to provide:

- **Bug fixes** — reproducible defects that affect correct behavior.
- **Security updates** — vulnerabilities are handled under the coordinated
  disclosure process described in [`SECURITY.md`](./SECURITY.md).
- **Documentation corrections** — inaccurate or missing documentation for
  existing behavior.

The following are **out of scope** for stable releases and will be deferred to
a future release:

- New features or behavioral changes.
- Performance improvements that require API or interface changes.
- Dependency upgrades that introduce breaking changes.

---

## How to obtain support

| Channel | Purpose |
|---|---|
| [GitHub Issues](https://github.com/Grid2op/grid2op/issues) | Bug reports and confirmed defects |
| [GitHub Discussions](https://github.com/orgs/Grid2op/discussions) | Questions, usage help, and general discussion |
| [Discord](https://discord.gg/cYsYrPT) | Real-time community chat |
| [Mailing list](https://lists.lfenergy.org/g/grid2op-discussion) | Announcements and broader community discussion |

For **security vulnerabilities**, do **not** open a public issue. Follow the
private disclosure process in [`SECURITY.md`](./SECURITY.md).

---

## Supported Python and dependency versions

Each release's `README` and release notes document the supported Python
versions (currently Python 3.9 – 3.13) and the compatible versions of key
dependencies (NumPy, PandaPower, lightsim2grid, pypowsybl2grid). Support for
older Python versions is dropped when they reach their upstream end-of-life.

---

## Commercial support

Grid2Op is a community-maintained open-source project. No commercial support
contracts are offered by the project itself. Organizations requiring guaranteed
service levels should consider engaging with contributors or LF Energy member
companies directly.

---

## Questions about this document

If anything here is unclear or out of date, please open an issue or a pull
request — improving this document is itself a valuable contribution.
