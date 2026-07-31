"""
Checks on the dependency extras declared in ``pyproject.toml``.

There are two parallel families of extras: pinned (``core``, ``classification``,
``tracking``, ``all``) and unpinned (the same names with a ``_latest`` suffix).
The unpinned family exists so that other packages can depend on ROICaT without
inheriting exact versions. Both families are hand-written, so they can drift
apart -- and a package silently missing from the unpinned family only shows up
as an ImportError in somebody else's install.

These tests compare package *names* only. Version constraints are what the two
families are supposed to differ on.
"""

from pathlib import Path
import re
import tomllib

import pytest


PATH_PYPROJECT = Path(__file__).resolve().parent.parent / 'pyproject.toml'

## Pinned extra -> its unpinned counterpart.
PAIRS_EXTRAS = {
    'core': 'core_latest',
    'classification': 'classification_latest',
    'tracking': 'tracking_latest',
    'all': 'all_latest',
    'dev': 'dev_latest',
}


def _load_extras():
    """
    Returns:
        (dict):
            ``{extra_name: [requirement_string, ...]}`` from ``pyproject.toml``.
    """
    if not PATH_PYPROJECT.exists():
        pytest.skip(f'pyproject.toml not found at {PATH_PYPROJECT}; not a source checkout.')
    with open(PATH_PYPROJECT, 'rb') as f:
        return tomllib.load(f)['project']['optional-dependencies']


def _resolve(extras, name, _seen=None):
    """
    Expands one extra into the set of distribution names it pulls in, following
    self-references like ``roicat[core]``.

    Args:
        extras (dict):
            All optional-dependency groups.
        name (str):
            The group to expand.

    Returns:
        (set):
            Lowercased distribution names, with ``-`` and ``_`` normalized, and
            without version constraints, extras or environment markers.
    """
    _seen = set() if _seen is None else _seen
    if name in _seen:
        return set()  ## Self-referential extras would otherwise recurse forever.
    _seen.add(name)

    out = set()
    for req in extras[name]:
        ## Drop the environment marker; it distinguishes platforms, not packages.
        req = req.split(';')[0].strip()
        match = re.match(r'^([A-Za-z0-9_.\-]+)(?:\[([^\]]*)\])?', req)
        dist, extras_requested = match.group(1), match.group(2)
        if dist.lower() == 'roicat':
            for sub in (extras_requested or '').split(','):
                out |= _resolve(extras, sub.strip(), _seen)
        else:
            out.add(dist.lower().replace('_', '-'))
    return out


@pytest.mark.parametrize('extra_pinned, extra_latest', sorted(PAIRS_EXTRAS.items()))
def test_pinned_and_latest_extras_name_the_same_packages(extra_pinned, extra_latest):
    """
    The unpinned family must cover exactly the same packages as the pinned one.
    """
    extras = _load_extras()
    for name in (extra_pinned, extra_latest):
        assert name in extras, f"pyproject.toml has no '{name}' extra."

    pkgs_pinned = _resolve(extras, extra_pinned)
    pkgs_latest = _resolve(extras, extra_latest)

    assert pkgs_pinned == pkgs_latest, (
        f"'{extra_pinned}' and '{extra_latest}' have drifted apart. "
        f"Only in '{extra_pinned}': {sorted(pkgs_pinned - pkgs_latest)}. "
        f"Only in '{extra_latest}': {sorted(pkgs_latest - pkgs_pinned)}."
    )


def test_latest_extras_are_unpinned():
    """
    A version constraint inside a ``_latest`` extra defeats its purpose: it is
    the constraint, not the package, that a downstream package cannot satisfy.
    """
    extras = _load_extras()
    constrained = {}
    for name in PAIRS_EXTRAS.values():
        for req in extras[name]:
            body = req.split(';')[0].strip()
            ## Strip the distribution name and any [extras] before looking for
            ## a constraint, so 'holoviews[recommended]' does not read as one.
            rest = re.sub(r'^[A-Za-z0-9_.\-]+(\[[^\]]*\])?', '', body).strip()
            if rest and not body.lower().startswith('roicat['):
                constrained.setdefault(name, []).append(req)
    assert not constrained, f'Version constraints found in unpinned extras: {constrained}'


def test_import_time_dependencies_are_declared():
    """
    Every package ``import roicat`` needs before any user code runs must be in
    the ``all`` extra. ``pandas`` was missing for a long time and worked only
    because seaborn happened to pull it in.
    """
    ## Third-party modules imported at module scope somewhere in the chain that
    ## `import roicat` walks, mapped to the distribution that provides them.
    MODULES_IMPORT_TIME = {
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'onnx': 'onnx',
        'onnxruntime': 'onnxruntime',
        'optuna': 'optuna',
        'pandas': 'pandas',
        'PIL': 'pillow',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'sparse': 'sparse',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'tqdm': 'tqdm',
        'yaml': 'pyyaml',
        'cv2': 'opencv-contrib-python-headless',
        'richfile': 'richfile',
    }
    extras = _load_extras()
    declared = _resolve(extras, 'all')
    missing = sorted({
        dist for dist in MODULES_IMPORT_TIME.values()
        if dist.replace('_', '-') not in declared
    })
    assert not missing, (
        f'Imported at import time but not declared in the "all" extra: {missing}'
    )
