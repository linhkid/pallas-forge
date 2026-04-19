# Publishing pallas-forge to PyPI

This guide walks through publishing `pallas-forge` to the Python Package Index so users can install it with a simple:

```bash
pip install pallas-forge
```

Two paths are documented:

1. **Automated (recommended)** — push a git tag, GitHub Actions builds and publishes via Trusted Publishing (OIDC). No API tokens to manage.
2. **Manual fallback** — build locally and upload with `twine`.

---

## Before your first release

### 1. Claim the name on PyPI and TestPyPI

PyPI names are first-come-first-served. Claim `pallas-forge` on both:

- **PyPI**: https://pypi.org/account/register/
- **TestPyPI** (staging): https://test.pypi.org/account/register/

You don't need to upload anything yet — just make sure the account exists and you can log in. Enable 2FA on both (PyPI requires it for maintainers).

### 2. Set up Trusted Publishing (OIDC)

This is the modern way to publish from CI. No tokens, no secrets.

**On PyPI** → your account → **Publishing** → *"Add a new pending publisher"*:

| Field | Value |
|---|---|
| PyPI Project Name | `pallas-forge` |
| Owner | `<your-github-username>` |
| Repository name | `pallas-forge` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Repeat on **TestPyPI** with environment name `testpypi`.

> "Pending publisher" means the PyPI project doesn't exist yet — once the first successful upload happens, it'll be created and linked.

### 3. Create GitHub environments

In your GitHub repo → **Settings** → **Environments** → create:

- `pypi` — optionally add a protection rule requiring approval before deploys
- `testpypi` — no protection needed; this is staging

### 4. Update the author URLs

In `pyproject.toml`, replace `nklinh91` in the `[project.urls]` block with your actual GitHub username if different. Also update the `authors` / `maintainers` blocks.

In `CHANGELOG.md`, update the `[Unreleased]` and `[0.1.0]` comparison URLs at the bottom.

In the `blog/` articles, do a find-and-replace on `<your-username>` and `<YOUR_USERNAME>`.

---

## The automated release workflow (recommended)

Once the one-time setup above is done, releases are a 4-step process.

### Step 1 — Update version + changelog

Edit `pallas_forge/_version.py`:
```python
__version__ = "0.2.0"   # bump according to semver
```

Also bump the `version` field in `pyproject.toml` to the same value.

Update `CHANGELOG.md`:
- Move items from `[Unreleased]` to a new `[0.2.0] — 2026-05-15` section
- Add comparison links at the bottom

Commit:
```bash
git add pallas_forge/_version.py pyproject.toml CHANGELOG.md
git commit -m "Release 0.2.0"
git push origin main
```

### Step 2 — (Optional) Test on TestPyPI first

For major changes, do a dry run against TestPyPI before the real release:

```
GitHub → Actions → "Publish to PyPI" → Run workflow → target: testpypi
```

Then verify the install works:

```bash
pip install -i https://test.pypi.org/simple/ pallas-forge==0.2.0
python -c "import pallas_forge; print(pallas_forge.__version__)"
```

### Step 3 — Tag and push

```bash
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0
```

The `v` prefix is required — the publish workflow triggers on `v*.*.*`.

### Step 4 — Watch CI

The `Publish to PyPI` workflow runs automatically. Watch it on the Actions tab. When it turns green:

- The package is live at https://pypi.org/project/pallas-forge/
- `pip install pallas-forge==0.2.0` works worldwide
- Create a GitHub Release from the tag (Releases → Draft a new release → Select tag → Generate release notes)

---

## Manual fallback (without CI)

If you need to publish without GitHub Actions (e.g. for a hotfix from a laptop):

### One-time setup

```bash
# Create TWO API tokens — one per index — scoped to pallas-forge
#   https://pypi.org/manage/account/token/
#   https://test.pypi.org/manage/account/token/

# Store in ~/.pypirc
cat > ~/.pypirc <<'EOF'
[distutils]
index-servers = pypi testpypi

[pypi]
  username = __token__
  password = pypi-YOUR_REAL_TOKEN_HERE

[testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-YOUR_TEST_TOKEN_HERE
EOF
chmod 600 ~/.pypirc
```

### Building and uploading

```bash
# From the repo root
conda activate pallas-forge

# Install build tooling
pip install -e ".[build]"

# Clean any stale artifacts
rm -rf dist/ build/ *.egg-info pallas_forge.egg-info

# Build sdist + wheel
python -m build

# Validate metadata (checks README renders, classifiers are valid, etc.)
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Verify the install in a fresh env
python -m venv /tmp/pftest && source /tmp/pftest/bin/activate
pip install -i https://test.pypi.org/simple/ pallas-forge
python -c "import pallas_forge; print(pallas_forge.__version__)"
deactivate

# If all good → upload to real PyPI
twine upload dist/*
```

---

## Troubleshooting

**"File already exists" on upload**
PyPI is append-only; you can never overwrite a version. Bump to the next patch (`0.2.1`) and republish.

**README doesn't render properly on PyPI**
PyPI renders a subset of GitHub-flavoured markdown. Check with:
```bash
twine check dist/*
```
Relative links (like `blog/01_why_pallas.md`) will appear as broken on the PyPI page — they only work on GitHub. If you want them clickable on PyPI, rewrite as absolute URLs (`https://github.com/<user>/pallas-forge/blob/main/blog/01_why_pallas.md`).

**Trusted publishing fails with "no matching publisher"**
The workflow name, repo, owner, and environment in your PyPI publisher config must match the workflow run *exactly*. The most common mistake is the environment name — make sure the `environment:` key in `publish.yml` matches what you registered on PyPI.

**`jax[tpu]` extra fails to resolve during install**
Expected on non-Linux platforms. `libtpu` is Linux-only. Users on macOS/Windows should install `pallas-forge` without the `tpu` extra and run in CPU interpret mode.

**`pip install pallas-forge` succeeds but imports fail**
Likely a wheel vs sdist issue. Check `dist/` — you should have both a `.tar.gz` and `.whl`. If only the sdist uploaded, `pip` might build from source and fail. Re-run `python -m build`.

---

## Version numbering

`pallas-forge` follows [SemVer](https://semver.org/):

- `0.x.y` — API may break between minors. Document breaking changes in the changelog.
- `1.x.y` — stable API. Breaking changes only in major bumps.
- Patch bumps (`0.1.1 → 0.1.2`) — bug fixes only; no API changes, no new deps.

Current: `0.1.0`. Expected path: `0.2.0` when the API crystallizes, `1.0.0` once external users are in production.

---

## Release checklist

Copy this for each release:

- [ ] All tests pass locally: `pytest tests/ -v`
- [ ] CI is green on `main`
- [ ] `pallas_forge/_version.py` bumped
- [ ] `pyproject.toml` version bumped to match
- [ ] `CHANGELOG.md` updated (move Unreleased → new section, update compare links)
- [ ] Tested on TestPyPI (optional for patch releases)
- [ ] Git tag pushed: `git tag -a vX.Y.Z -m "Release X.Y.Z" && git push origin vX.Y.Z`
- [ ] Publish workflow green on Actions tab
- [ ] PyPI page reachable: `https://pypi.org/project/pallas-forge/X.Y.Z/`
- [ ] `pip install pallas-forge==X.Y.Z` works in a fresh venv
- [ ] GitHub Release created from the tag with generated notes
