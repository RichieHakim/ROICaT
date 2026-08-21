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

    The list below describes the *current* import structure, in which
    ``roicat/__init__.py`` eagerly imports every submodule. If those imports are
    ever made lazy, fewer packages will be needed at import time and entries
    here should be removed -- a failure after such a change means the list is
    stale, not that the change was wrong.
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


######################################################################################################################################
######################################################## PYTHON VERSIONS #############################################################
######################################################################################################################################

PATH_BUILD_YML = Path(__file__).resolve().parent.parent / '.github' / 'workflows' / 'build.yml'

## Candidate minor versions to ask `requires-python` about. Wide enough that the
## range does not need revisiting when Python moves on.
VERSIONS_CANDIDATE = [f'3.{n}' for n in range(8, 30)]


def _versions_supported():
    """
    Returns:
        (set): the ``3.x`` strings that ``requires-python`` admits.
    """
    from packaging.specifiers import SpecifierSet

    if not PATH_PYPROJECT.exists():
        pytest.skip(f'pyproject.toml not found at {PATH_PYPROJECT}; not a source checkout.')
    with open(PATH_PYPROJECT, 'rb') as f:
        spec = SpecifierSet(tomllib.load(f)['project']['requires-python'])
    return {v for v in VERSIONS_CANDIDATE if spec.contains(v)}


def _versions_tested():
    """
    Returns:
        (set): the ``3.x`` strings in the CI build matrix. Commented-out entries
        are not included -- YAML drops them, which is the intended reading.
    """
    import yaml

    if not PATH_BUILD_YML.exists():
        pytest.skip(f'build.yml not found at {PATH_BUILD_YML}; not a source checkout.')
    with open(PATH_BUILD_YML, 'r') as f:
        workflow = yaml.safe_load(f)
    return {str(v) for v in workflow['jobs']['build']['strategy']['matrix']['python-version']}


def test_ci_tests_every_supported_python():
    """
    The versions CI runs and the versions ``requires-python`` admits must match.

    Drift in either direction is a problem. A version admitted but not tested is
    a support claim with nothing behind it; a version tested but not admitted
    means the install step in that job cannot have run at all, so the job proves
    nothing while still reporting green. 3.13 spent months commented out of the
    matrix while the README said it was unsupported and nothing checked either.
    """
    supported, tested = _versions_supported(), _versions_tested()
    assert supported == tested, (
        f"requires-python admits {sorted(supported)} but CI runs {sorted(tested)}. "
        f"Only in requires-python: {sorted(supported - tested)}. "
        f"Only in CI: {sorted(tested - supported)}."
    )


def test_readme_states_the_supported_pythons():
    """
    The README's Requirements line is what users actually read, so it must name
    every supported version and nothing else.
    """
    path_readme = Path(__file__).resolve().parent.parent / 'README.md'
    if not path_readme.exists():
        pytest.skip(f'README.md not found at {path_readme}; not a source checkout.')
    line = next(
        (l for l in path_readme.read_text(encoding='utf-8').splitlines() if '**Python' in l),
        None,
    )
    assert line is not None, "No '**Python ...**' requirements line found in README.md."

    named = set(re.findall(r'3\.\d+', line))
    assert named == _versions_supported(), (
        f"README requirements line names {sorted(named)}: {line.strip()!r}, "
        f"but requires-python admits {sorted(_versions_supported())}."
    )


## Packages that *are* a notebook environment: a kernel to execute cells, or a
## server/UI to open them in. ROICaT deliberately ships none of these, because
## notebook setups vary too much for one choice to suit everyone -- users bring
## their own.
##
## The line is between an environment and a library. Rendering packages are not
## on this list even when their names say "jupyter": `jupyter-bokeh` is part of
## ROICaT's plotting stack (it is what renders bokeh output as a widget), so it
## is a declared dependency like bokeh and holoviews. What it pulls in
## transitively -- ipywidgets, and IPython -- is a consequence of that choice,
## not a notebook environment ROICaT is providing.
DISTS_NOTEBOOK_ENVIRONMENT = {
    'jupyter',
    'jupyterlab',
    'notebook',
    'jupyter-server',
    'ipykernel',
}


@pytest.mark.parametrize('extra', sorted(set(PAIRS_EXTRAS) | set(PAIRS_EXTRAS.values())))
def test_no_extra_declares_a_notebook_environment(extra):
    """
    No extra should install a notebook stack.

    `jupyter` sat in the `dev` extra used by nothing: `tests/test_interactive.py`
    drives bokeh's own server via selenium, not a kernel. Meanwhile no extra
    provided an actual kernel or server, so following the README literally left a
    user unable to open the notebooks it points them at -- enough notebook
    machinery to be load-bearing by accident, not enough to be useful.
    """
    extras = _load_extras()
    declared = _resolve(extras, extra)
    offenders = sorted(declared & DISTS_NOTEBOOK_ENVIRONMENT)
    assert not offenders, (
        f"extra '{extra}' declares notebook-environment packages {offenders}. "
        'ROICaT does not provide a notebook environment; users install their own.'
    )


@pytest.mark.parametrize('extra', ['classification', 'classification_latest', 'all', 'all_latest'])
def test_classification_declares_the_bokeh_widget_renderer(extra):
    """
    `jupyter-bokeh` must stay declared wherever the drawing notebook's plots are.

    Its name makes it look like notebook-environment tooling, and it was removed
    once on that basis. This pins why it is not.
    `visualization.select_region_scatterPlot` calls `hv.extension('bokeh')`,
    holoviews calls `pn.extension()` (holoviews/util/__init__.py), and panel's
    `config._detect_comms` imports `jupyter_bokeh` when it detects Colab or
    VSCode -- warning that interactive output will not render without it. In
    plain JupyterLab it genuinely is not needed, but
    `A1_classify_by_drawingSelection.ipynb` carries a Colab badge in the README
    and installs `roicat[classification]` in its own Colab cell, so dropping it
    breaks a workflow ROICaT advertises.
    """
    extras = _load_extras()
    assert 'jupyter-bokeh' in _resolve(extras, extra), (
        f"extra '{extra}' no longer declares jupyter-bokeh; the drawing notebook "
        'will not render interactively in Colab or VSCode.'
    )


def test_documented_notebook_paths_exist():
    """
    Every notebook path named in the README, the docs, or a notebook's own
    cross-links must exist.

    The README told users to `cd` into `ROICaT/notebooks/jupyter`, a directory
    deleted in 790ba8a, and several notebooks linked to a
    `demo_custom_data_importing.ipynb` that has never existed. Both are invisible
    to every other test, because nothing imports a markdown link.
    """
    root = PATH_PYPROJECT.parent
    if not (root / 'notebooks').is_dir():
        pytest.skip('notebooks/ not present; not a source checkout.')

    sources = [root / 'README.md']
    sources += sorted((root / 'docs').rglob('*.rst')) if (root / 'docs').is_dir() else []
    sources += sorted((root / 'notebooks').rglob('*.ipynb'))

    missing = []
    for path in sources:
        text = path.read_text(encoding='utf-8', errors='ignore')
        ## Any reference to a path under notebooks/, with or without a URL prefix.
        for ref in set(re.findall(r'notebooks/[A-Za-z0-9_./\- ]*?\.ipynb', text)):
            if not (root / ref).exists():
                missing.append(f'{path.relative_to(root)} -> {ref}')
    assert not missing, f'Documented notebook paths that do not exist: {sorted(missing)}'
