# Grid2Op — Governance Document

*[LF Energy Foundation](https://lfenergy.org/)*

> **NOTE:** This document provides the governance structure for the Grid2Op
> project hosted by LF Energy Foundation. All participants are bound by the
> project's [Code of Conduct](https://github.com/Grid2op/grid2op/blob/master/CODE_OF_CONDUCT.md).

---

## 1. Overview

This project aims to be governed in a transparent, accessible way for the
benefit of the community. All participation in this project is open and not
bound to corporate affiliation. Participants are bound by the project's
[Code of Conduct](https://github.com/Grid2op/grid2op/blob/master/CODE_OF_CONDUCT.md).

---

## 2. Ecosystem Structure

The Grid2Op ecosystem is organized into two categories of packages:

### 2.1 Core Packages

Core packages are well-maintained, well-documented, and extensively tested.
They form the foundation of the Grid2Op ecosystem that the community can rely
on and reuse. The current Core packages are:

- grid2op
- lightsim2grid

Core packages follow the full governance process described in this document.
All Core repositories fulfill the OpenSSF Best Practices badge requirements.
Unless stated otherwise, each "Core" package is independently versioned and released.

### 2.2 Example / Companion Packages

Example packages are experimental or companion contributions that demonstrate
usage of Core packages. They may include:

- Computation engines (e.g. pypowsybl2grid)
- Baselines solving L2RPN competitions
- Example UX or HMI implementations
- Companion packages (e.g. chronix2grid for data generation)
- External contributions from research papers

Example packages have lighter governance requirements, acknowledging their
experimental nature. Contributors and users should be aware that these packages
may not be maintained to the same standard as Core packages.

### 2.3 Archive Policy

An Example package that is no longer maintained may be archived by a TSC
majority vote, or upon request by the project Lead. When a repository is
archived:

- The repository description is updated to explicitly note that the project is
  not maintained.
- The archival date is recorded.
- The last known compatible versions of Core packages are documented.

### 2.4 Adoption into Core

An Example package that has matured — demonstrating sustained maintenance,
broad community use, and alignment with the project's quality standards — may
be adopted into Core. This process is intentionally lightweight to encourage
contribution while ensuring the quality bar for Core packages is upheld.

**Criteria for adoption**

A package may be considered for adoption when it meets all of the following:

- It is actively maintained, with timely responses to issues and pull requests.
- It has comprehensive documentation and test coverage comparable to existing
  Core packages.
- It fulfills, or has a credible short-term plan to fulfill, the OpenSSF Best
  Practices badge requirements.
- It is demonstrably used by members of the community beyond its original
  authors.
- Its maintainers are willing to commit to the Committer responsibilities
  defined in Section 4.2.

**Adoption process**

- The package maintainer (or any TSC member on their behalf) opens a GitHub
  issue in the project's governance repository proposing the adoption and
  summarising how the criteria above are met.
- The TSC reviews the proposal, inviting the maintainer to present at a TSC
  meeting if needed.
- Adoption is approved by a majority vote of the TSC.
- Upon approval, the repository is moved under the Core governance process: it
  is added to the list of Core packages in this document, its committers are
  added to `COMMITTERS.csv`, and it inherits all Core requirements (OpenSSF
  badge, SBOM on release, SCA scans, etc.).

> **Note:** There is no automatic adoption route. A package remaining in the
> Example category indefinitely is perfectly acceptable; adoption should only
> happen when the community and the maintainers are both ready.

---

## 3. OpenSSF Best Practices Badge

Grid2Op maintains a single OpenSSF Best Practices badge covering the entire
Grid2Op organization. 

Each Core repository is expected to fulfill the badge
requirements. 

Example / Companion repositories are encouraged to inherit processes and
policies from the Core project, but are not strictly required to do so in
early stages of development.

---

## 4. Project Roles

### 4.1 Contributor

The contributor role is the starting role for anyone participating in the
project and wishing to contribute code.

**Process for becoming a Contributor**

- Review the Contribution Guidelines to ensure your contribution is in line
  with the project's coding and styling guidelines.
- Submit your code as a Pull Request with the appropriate
  [Developer Certificate of Origin (DCO)](https://developercertificate.org/)
  sign-off.
- Have your submission approved by the committer(s) and merged into the
  codebase.

### 4.2 Committer

The committer role enables the contributor to commit code directly to the
core repositories, and comes with the responsibility of being a responsible leader in
the community.

**Process for becoming a Committer**

- Show your experience with the codebase through contributions and engagement
  on Grid2Op community channels.
- Request to become a committer by creating a new Pull Request that adds your
  name and details to the Committers file, and request existing committers to
  approve.
- After a majority of committers approve, merge the PR and tag whoever manages
  GitHub permissions to update the committers team.

**Committer Responsibilities**

- Monitor email aliases (if any).
- Monitor Grid2Op communication channels (Slack, Discord — delayed response is
  acceptable).
- Triage GitHub issues and perform pull request reviews for other committers
  and the community.
- Ensure ongoing PRs are moving forward at the right pace or are closed.
- In general, be willing to spend at least 20% of one's time working on the
  project (~1 business days per week).

**Loss of Committer Status**

If a committer is no longer interested or cannot perform the duties listed
above, they should volunteer to be moved to emeritus status. In extreme cases
this can also occur by a vote of the committers per the voting process below.

### 4.3 Technical Steering Committee (TSC)

The TSC acts as the guiding committee for the Grid2Op LF Energy Foundation
community, providing governance and oversight, and ensuring that the project
adheres to all of LF's guidelines and stipulations. Serving on the TSC is a
distinct role from the Committer role, but community members may serve in both
roles simultaneously.

**Process for becoming a TSC Member**

- Show your experience and engagement in the broader AI and energy community,
  and with the Grid2Op community specifically via contributions and engagement
  on Grid2Op community channels.
- Express your interest by opening a GitHub issue in the project's governance
  repository requesting TSC membership.
- New TSC members will be confirmed by a majority vote of the existing TSC
  members.

**TSC Member Responsibilities**

- Attend periodic meetings, held likely monthly but at least quarterly.
- Actively participate in discussions and activities surrounding project
  governance, oversight, milestones, and monitoring — including synchronous
  participation during meetings and asynchronous participation in between
  meetings.
- TSC members are expected to contribute on a best-effort basis; there is no
  fixed time commitment, but consistent engagement is expected.

**Loss of TSC Status**

If a TSC member is no longer interested or cannot perform the TSC member
duties listed above, they should volunteer to be moved to emeritus status. In
extreme cases this can also occur by a vote of the remaining TSC members per
the voting process below.

### 4.4 Lead

The TSC members will elect a Lead (and optionally a Co-Lead) who will be the
primary point of contact for the project and representative to the TAC. The
Lead is responsible for the overall project health and direction, coordination
of activities, and working with other projects and committees for the continued
growth of the project.

The Lead role has no fixed term. Re-election or replacement occurs if the
current Lead steps down, is no longer able to fulfill the role, or upon a
majority vote of the TSC.

### 4.5 Collaborator Review Policy

**Review Process**

Before any contributor is granted escalated permissions, the following steps
must be completed:

- **Identity verification.** The contributor's identity must be traceable to a
  known, trusted entity. Acceptable evidence includes: a verifiable
  organizational email address; a consistent history of DCO-signed commits
  under a stable identity; or known association with an organization already
  active in the Grid2Op community.
- **Contribution history review.** The existing committers and/or TSC must
  review the candidate's prior contributions for code quality, responsiveness
  to review, and alignment with project values.
- **Explicit approval vote.** A majority vote of the existing committers (for
  committer access) or TSC members (for TSC-level or infrastructure access) is
  required before permissions are granted.
- **Least-privilege assignment.** Permissions are scoped to the minimum
  necessary. Repository write access, merge rights, and secret access are
  granted separately and only as each becomes justified.
- **Recorded decision.** The granting of escalated permissions must be
  documented — typically via the merged pull request that adds the person to
  `COMMITTERS.csv` or an equivalent record — so the decision is auditable.

**Secrets and Infrastructure Access**

Access to project secrets (PyPI tokens, CI signing keys, cloud infrastructure)
is restricted to a named subset of committers. Granting or revoking this
access requires TSC approval and must be logged in the project's internal
access register.

**Periodic Review**

The TSC reviews the list of active committers and their permission levels at
least once per year. Committers who have been inactive for 12 months or more
will be contacted; absent a response, they will be moved to emeritus status
and their permissions revoked.

---

## 5. Release Process

Project releases are made on a rolling/ad hoc basis, as agreed upon by the
committers based on the readiness of features and fixes. There is no fixed
release cadence.

The project follows Semantic Versioning (`major.minor.patch`):

- **MAJOR** — incompatible API changes.
- **MINOR** — new backward-compatible functionality.
- **PATCH** — backward-compatible bug fixes.

Release criteria, changelogs, tagging procedures, and per-repository release
checklists (including supply-chain requirements such as Software Bills of
Materials) are documented in the `RELEASE` file of the respective repository.

Each Core repository must pass a Software Composition Analysis (SCA) scan
with no unaddressed critical or high-severity findings prior to any release.

---

## 6. Conflict Resolution and Voting

In general, we prefer that technical issues, committer membership, and TSC
membership are amicably worked out between the persons involved. If a dispute
cannot be decided independently, the committers and/or TSC members can be
called in to decide an issue. If the committers and/or TSC members themselves
cannot decide an issue, the issue will be resolved by voting.

The voting process is a simple majority in which each committer and/or TSC
member receives one vote.

---

## 7. Code of Conduct

This project follows the
[LF Energy Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).
All participants are expected to uphold this code. 

Please report unacceptable
behavior to the project committers or TSC.

---

## 8. Communication

This project, like all open source, is a global community. In addition to the
Code of Conduct, this project will:

- Keep all communication on open channels (see below).
- Be respectful of time and language differences between community members
  (e.g. when scheduling meetings, for email/issue responsiveness).
- Ensure tools are able to be used by community members regardless of their
  region.

### 8.1 Communication Channels

- **Discord:** https://discord.gg/cYsYrPT
- **Mailing list:** https://lists.lfenergy.org/g/grid2op-discussion
- **Slack:** LF Energy Slack workspace, `#grid2op` channel
- **GitHub Discussions:** https://github.com/rte-france/Grid2Op/discussions

If you have concerns about communication challenges for this project, please
contact the committers.

---

*Last updated: May 2026*