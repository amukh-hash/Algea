# Ops Troubleshooting

## Windows: HOME / Playwright setup
- PowerShell (current session):
  - `$env:HOME=$env:USERPROFILE`
- Permanent user env var:
  - `setx HOME "%USERPROFILE%"`
- Optional browser cache path:
  - `setx PLAYWRIGHT_BROWSERS_PATH "%USERPROFILE%\\ms-playwright"`

## Stale process on a port
- Find process:
  - `netstat -ano | findstr :8000`
- Kill process:
  - `taskkill /PID <pid> /F`

## API timeout symptoms
- `/api/orchestrator/status` returns 504 when orchestrator probe exceeds timeout.
- Check `X-Request-ID` in response headers for trace correlation.

## npm registry access for lint dependencies
If `npm run lint` fails with missing `eslint`/`eslint-config-next`, verify your registry credentials and scope mapping.

Example `frontend/.npmrc` for npmjs:

```ini
registry=https://registry.npmjs.org/
always-auth=true
//registry.npmjs.org/:_authToken=${NPM_TOKEN}
```

If your org uses an internal mirror, point `registry=` to that endpoint and configure auth accordingly.

### Why `lint:maybe` can skip in restricted environments
`npm run lint:maybe` is designed for sandboxed/restricted environments where devDependencies cannot be installed (e.g., HTTP 403 from registry). It prints a clear skip message and exits 0 instead of failing the whole run. In CI and developer machines with full access, use `npm run lint:ci`.

## Local commands to fully enable lint and lockfiles
Run the following from repo root:

```bash
cd frontend
npm install
npm run lint:ci
npm run test
git add package-lock.json package.json .eslintrc.json
git commit -m "Add ESLint deps and CI-safe lint scripts"
```
