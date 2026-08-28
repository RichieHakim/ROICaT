# Release workflow failure, 2026-08-28: Sigstore TUF root rotation

Run: <https://github.com/RichieHakim/ROICaT/actions/runs/33207812352>
Workflow: `.github/workflows/pypi_release.yml` ("Publish Python 🐍 distribution 📦 to PyPI")
Commit: `102d1ae1` on `main`, version `1.7.8`, triggered manually via `workflow_dispatch`.

## What failed

Two of the three jobs succeeded. The `build` job produced the wheel and sdist,
and `publish-to-pypi` uploaded them. The third job, `github-release`, failed at
its first real step, `Sign the dists with Sigstore`. Every step after it was
skipped, including `Create GitHub Release`.

```
tuf.api.exceptions.UnsignedMetadataError: root was signed by 0/3 keys
```

The traceback runs `sigstore/_cli.py:_sign` → `SigningContext.production()` →
`TrustedRoot.production()` → `TrustUpdater(...)` → python-tuf's
`Updater.__init__` → `TrustedMetadataSet._load_trusted_root`. The failure is at
the very first thing Sigstore does: bootstrapping its root of trust, before any
signing is attempted.

## Why

`pypi_release.yml` line 74 pins the action but not what the action installs:

```yaml
    - name: Sign the dists with Sigstore
      uses: sigstore/gh-action-sigstore-python@v3.0.0
```

Action `v3.0.0` was published on 2024-07-15. Its `requirements.txt` contains a
floating range, not a pin:

```
sigstore ~= 3.0
```

In this run that resolved to **sigstore-python 3.6.7** with **tuf 6.0.0** (see
log lines 148-176 of the signing job). sigstore-python 3.6.7 ships an embedded
copy of the Sigstore TUF root, used to bootstrap trust on a machine with no
cached metadata — which is every fresh GitHub runner.

Sigstore rotated that TUF root in late August 2026. The root embedded in the
3.x line is now superseded and no longer verifies, which is what "signed by 0/3
keys" means: three signatures were required and none validated.

Two facts confirm this is an external change and not something in this repo:

1. **The workflow file has not changed.** `git diff 17a8e85a 102d1ae1 --
   .github/workflows/pypi_release.yml` is empty. The same file, with the same
   `v3.0.0` pin, succeeded on 2026-08-03 for v1.7.7.
2. **Other projects broke at the same moment.** A GitHub-wide search for the
   error string returns a cluster of fixes dated 2026-08-27 and 2026-08-28,
   including one titled "pin the Sigstore action past the TUF root rotation".
   Every one of them was on `v3.0.0` and every one moved to `v3.5.0`.

The underlying fragility is that action `v3.0.0` does not pin its Python
dependencies, so the same workflow file installs different code on different
days. It was never reproducible; the rotation is just what finally exposed that.

## Consequences to be aware of

The release is half-complete:

- `roicat==1.7.8` **is published on PyPI**, uploaded 2026-08-28T20:21:16 during
  this run. Users can install it.
- There is **no `v1.7.8` GitHub Release and no `v1.7.8` git tag**. The newest
  release and tag are both `v1.7.7`. The `Create GitHub Release` step never ran.
- There are **no Sigstore signature bundles** for the 1.7.8 artifacts.

Re-running the workflow as-is will fail again at the same step, and the PyPI
upload will additionally fail because 1.7.8 already exists and PyPI does not
allow re-uploading a version.

## The fix

One line in `.github/workflows/pypi_release.yml`:

```diff
-      uses: sigstore/gh-action-sigstore-python@v3.0.0
+      uses: sigstore/gh-action-sigstore-python@v3.5.0
```

Action `v3.5.0` (published 2026-07-29) fully pins its dependencies with hashes
in `requirements/main.txt`: `sigstore==4.5.0`, `tuf==7.0.0`. sigstore-python
4.5.0 ships the current root, and because the pin is exact the action will keep
installing the same code rather than drifting again.

For a stronger version, pin to the commit rather than the tag, so a moved tag
cannot change what runs:

```yaml
      uses: sigstore/gh-action-sigstore-python@790bc6befb9d733738f18d8f895854b453640ec9 # v3.5.0
```

### Finishing the 1.7.8 release

The version is already on PyPI, so the remaining gap is the tag, the GitHub
Release and the signatures. The simplest repair is to create the tag and
release directly against `102d1ae1` rather than re-running the workflow:

```bash
git tag v1.7.8 102d1ae1 && git push origin v1.7.8
gh release create v1.7.8 --title v1.7.8 \
  --notes "Release for version 1.7.8. Also available on PyPI: https://pypi.org/project/roicat/1.7.8/"
```

Signatures for 1.7.8 would have to be produced and attached separately, or
simply skipped, with signing resuming from 1.7.9 once the action is upgraded.

## Worth considering separately

The `github-release` job depends on `publish-to-pypi`, and PyPI uploads cannot
be undone or repeated. Any failure after the upload leaves exactly the state
seen here: a published version with no tag and no release. Moving the tag and
release creation ahead of the PyPI upload, or making the signing step
non-blocking with `continue-on-error: true`, would keep one flaky external
service from splitting a release in half.

The three "List files in ..." steps (lines 79-87) are debugging leftovers and do
nothing for the release.
