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

## Supported Versions

Security fixes are made against the latest released minor version. Older
versions are not patched.
