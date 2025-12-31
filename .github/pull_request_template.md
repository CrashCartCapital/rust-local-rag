## AI Change Review Checklist

> **Note**: This checklist is required for all AI-assisted changes.

### 1. Verification
- [ ] **Run Standard Checks**: Ran `make ci` (includes `check`, `lint`, `test`, `build`).
  - Output: (Paste relevant output or "Passed")
- [ ] **Manual Testing**: Verified the change manually (if applicable).
  - Describe what was tested: _________________

### 2. Documentation
- [ ] **Update Docs**: Updated `CLAUDE.md`, `README.md`, or code comments if behavior changed.
- [ ] **New Dependencies**: If added, verified license and updated `Cargo.toml`.

### 3. Risk Assessment
- [ ] **Edge Cases**: Considered empty inputs, errors, and concurrency issues.
- [ ] **Migrations**: Noted any env vars, schema changes, or config updates required.
- [ ] **Security**: No sensitive data exposed; `unsafe` code minimized and justified.

### 4. What I did NOT verify (REQUIRED)
> explicitly state what was out of scope or skipped
- [ ] I did NOT verify: _________________

---

## Description
<!-- Describe your changes here -->

## Related Issues
<!-- Link to related issues -->
