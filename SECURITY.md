# Security Policy

Please report any **critical** or **important** security vulnerability, suspected or confirmed, privately to the grid2op maintainer, currently:

- [Benjamin DONNOT](mailto:benjamin.donnot@rte-france.com)

In your e-mail, please provide basic information about who you are (name and company or research group) as well as detailed steps to reproduce the vulnerabilities (python code, screenshots etc.) and the
effect of said vulnerabilities.

For *moderate* or *low-severity* security vulnerabilities, you can use either :
- use the public [Github issues](https://github.com/Grid2op/grid2op/issues)
- report them via the grid2op discord server (https://discord.gg/cYsYrPT)
- send an e mail to one of the above mentionned person.

In order to help you assess the severity of the potential vulnerability, you can use the [Apache severity rating](https://security.apache.org/blog/severityrating/).

If you are not sure whether the issue should be reported privately or publicly, please make a private report.

## Supported version

**Critical** vulnerabilities will be backward implemented for all patches of the last minor release within the previous calendar year. For example, if a critical vulnerability impact grid2op 1.10.4 
(major release 1, minor release 10, patch release 4) the security patch will be made for all patches of grid2op concerning 1.10 (so 1.10.0, 1.10.1, 1.10.2, 1.10.3) as well as all the patch releases
of the current "release train".

**Critical** and **important** vulnerabilities will be forward implemented for all patches of the last published minor release within the previous calendar year. For example, if the current grid2op
relase is 1.11.1 (major release 1, minor release 11, patch release 1) then version 1.11.0 and 1.11.1 will be patched.

For critical, important and moderate patches will be applied for the last patch release of grid2op.

Security patches of all level of severity will be implemented in the current "release train", available in the next release.
