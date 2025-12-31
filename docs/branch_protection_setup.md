# Branch Protection Setup

To prevent chaos and ensure code quality, configure the following branch protections for your main branch (e.g., `main` or `master`).

## 1. Access Branch Protection Settings

1.  Go to your repository on GitHub.
2.  Click **Settings** > **Branches**.
3.  Click **Add branch protection rule**.
4.  **Branch name pattern**: `main` (or `master`)

## 2. Configure Rules

Check the following options:

### 🛡️ Require a pull request before merging
*   **Require approvals**: Recommended (e.g., 1).
*   **Dismiss stale pull request approvals when new commits are pushed**: Recommended.

### 🛡️ Require status checks to pass before merging
*   **Require branches to be up to date before merging**: Recommended.
*   **Status checks that are required**:
    *   Search for and select: `ci` (This corresponds to the job name in `.github/workflows/ci.yml`)

### 🛡️ Do not allow bypassing the above settings
*   Check this to ensure these rules apply to administrators as well (optional but recommended for strict hygiene).

## 3. Save
Click **Create** or **Save changes**.

---

## Canonical Commands

Always verify your code locally before pushing to ensure these checks pass:

*   **Format**: `make fmt`
*   **Lint**: `make clippy` (or `make lint`)
*   **Test**: `make test`
*   **Full Check**: `make ci` (runs all the above + build)
