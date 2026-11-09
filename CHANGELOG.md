# Changelog

## [Unreleased]

### Fixed

- Use valid pandas boolean masks in quality metrics and include `cobble` in
  the class distribution, with regression tests for labels, confidence bands,
  missing columns, empty inputs, and persisted JSON output.

### Security

- Bind the standalone review labeler to `127.0.0.1` by default and refuse
  non-loopback startup unless `COASTPILOT_LABEL_TOKEN` is non-blank.
- Add regression tests for safe startup defaults, rejected unauthenticated
  network startup, and token-protected read/write routes using temporary CSVs.
- Document SSH-tunnel access and the token requirement for alternative launchers.
