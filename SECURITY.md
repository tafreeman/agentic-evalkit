# Security Policy

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub
issues, discussions, or pull requests.

Instead, report vulnerabilities privately using
[GitHub's private security advisory workflow](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
for this repository ("Security" tab → "Report a vulnerability").

Include as much detail as you can:

- a description of the vulnerability and its potential impact;
- steps to reproduce it, including affected version(s);
- any known mitigations.

## Scope

This policy covers the `agentic-evalkit` package itself. Vulnerabilities in
systems evaluated *through* `agentic-evalkit` (targets under test, such as
ARP or ExecutionKit deployments) are out of scope for this repository and
should be reported to the maintainers of those systems.

`agentic-evalkit` separates datasets, grading, and reporting from the
system under test through callable/subprocess/HTTP targets; it never
imports ARP, `agentic-tools`, or ExecutionKit internals (see
[ADR-0001](docs/adr/0001-standalone-boundary.md)). Legacy evaluation code
that remains in a host repository is likewise out of scope here — this
package neither imports nor migrates it, so a vulnerability in that
legacy code should be reported to the host repository's own security
policy, not this one.

## Supported Versions

Security fixes are made against the latest released minor version. Older
versions are not patched.
