---
# Codacy Rules
Configuration for AI behavior when interacting with Codacy's MCP Server

## After ANY successful file edit
- YOU MUST run the `codacy_cli_analyze` tool for each edited file
- If any issues are found, propose and apply fixes immediately
- This is a critical requirement, not optional

## After ANY dependency installation
- IMMEDIATELY run `codacy_cli_analyze` with tool set to "trivy"
- Check for security vulnerabilities in new packages
- Stop all operations if vulnerabilities found
- Fix security issues before continuing

## Repository setup
- Use `git remote -v` to determine provider/organization/repository
- Provider mapping: GitHub = "gh", Bitbucket = "bb", GitLab = "gl"
- If repository not found in Codacy, offer to run `codacy_setup_repository`
---
