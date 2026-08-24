"""
Checks on the dependency declaration in ``pyproject.toml``.

Everything ROICaT imports is declared in the ``core`` extra, so
``pip install roicat[core]`` is a complete install. ``latest`` names the same
packages unconstrained, for downstream packages that cannot inherit pins.

Three things can quietly break that: a new ``import`` of something nobody
declared (it works locally because some other package happened to pull it in),
the two families drifting apart, and a version constraint creeping into
``latest``. All three are tested here.
"""

from typing import Dict, List, Set  ## typing

import ast
from pathlib import Path
import re
import sys
import tomllib  ## built-ins

import pytest  ## third-party


PATH_ROOT = Path(__file__).resolve().parent.parent
PATH_PYPROJECT = PATH_ROOT / 'pyproject.toml'
DIR_PACKAGE = PATH_ROOT / 'roicat'

## Names that resolve to `core` or `latest` rather than holding a list of their
## own. `all` is the documented install command; the rest are kept because they
## are printed in older docs, papers and downstream scripts.
NAMES_EXTRAS_LEGACY = [
    'classification',
    'tracking',
    'all',
    'pinned',
    'core_latest',
    'classification_latest',
    'tracking_latest',
    'all_latest',
    'dev_pinned',
]

## Pinned extra -> its unconstrained counterpart.
PAIRS_EXTRAS = {
    'core': 'latest',
    'dev': 'dev_latest',
}

## Importable module name -> the distribution on PyPI that provides it, for the
## cases where the two differ.
DISTS_BY_MODULE = {
    'cpuinfo': 'py-cpuinfo',
    'cv2': 'opencv-contrib-python-headless',
    'fast_hdbscan': 'fast-hdbscan',
    'IPython': 'ipython',
    'PIL': 'pillow',
    'romatch': 'romatch-roicat',
    'skimage': 'scikit-image',
    'sklearn': 'scikit-learn',
    'umap': 'umap-learn',
    'yaml': 'pyyaml',
}

## Modules imported inside ROICaT that are deliberately not base dependencies.
## Each is reached only by opting into it, so an install without it is a
## working install.
MODULES_NOT_DEPENDENCIES = {
    ## Not a distribution at all: a .py file downloaded alongside the ROInet
    ## weights and imported after its directory is added to sys.path.
    'model',
    ## Ships in romatch-roicat's 'fused-local-corr' extra, which has x86_64
    ## Linux wheels only. RoMa runs without it, more slowly.
    'local_corr',
    ## The original HDBSCAN, offered as an alternative clusterer. Needs a C++
    ## toolchain to build on Windows; fast-hdbscan is the default and is pure
    ## Python plus numba.
    'hdbscan',
    ## An alternative parallel backend, used only when a caller passes
    ## method='mpire' to roicat.helpers.map_parallel.
    'mpire',
    ## Training-run logging, reached only from roicat/model_training/. Declared
    ## in the `training` extra.
    'wandb',
}


def _load_pyproject() -> dict:
    """
    Returns:
        (dict):
            Parsed ``pyproject.toml``.
    """
    if not PATH_PYPROJECT.exists():
        pytest.skip(f'pyproject.toml not found at {PATH_PYPROJECT}; not a source checkout.')
    with open(PATH_PYPROJECT, 'rb') as f:
        return tomllib.load(f)


def _name_distribution(requirement: str) -> str:
    """
    Pulls the distribution name out of one requirement string, discarding the
    version constraint, any ``[extras]`` and any environment marker.

    Args:
        requirement (str):
            A PEP 508 requirement, e.g. ``"holoviews[recommended]==1.23.1"``.

    Returns:
        (str):
            The distribution name, lowercased with ``_`` normalized to ``-``.
    """
    body = requirement.split(';')[0].strip()
    name = re.match(r'^([A-Za-z0-9_.\-]+)', body).group(1)
    return name.lower().replace('_', '-')


def _names_declared(requirements: List[str]) -> Set[str]:
    """
    Args:
        requirements (List[str]):
            PEP 508 requirement strings.

    Returns:
        (Set[str]):
            Their normalized distribution names.
    """
    return {_name_distribution(req) for req in requirements}


def _resolve(extras: dict, name: str, _seen: Set[str] = None) -> Set[str]:
    """
    Expands one extra into the distributions it pulls in, following
    self-references like ``roicat[core]``.

    Args:
        extras (dict):
            All optional-dependency groups.
        name (str):
            The group to expand.

    Returns:
        (Set[str]):
            Normalized distribution names, without version constraints, extras
            or environment markers.
    """
    _seen = set() if _seen is None else _seen
    if name in _seen:
        return set()  ## Self-referential extras would otherwise recurse forever.
    _seen.add(name)

    out = set()
    for req in extras[name]:
        body = req.split(';')[0].strip()
        match = re.match(r'^([A-Za-z0-9_.\-]+)(?:\[([^\]]*)\])?', body)
        dist, extras_requested = match.group(1), match.group(2)
        if dist.lower() == 'roicat':
            for sub in (extras_requested or '').split(','):
                out |= _resolve(extras, sub.strip(), _seen)
        else:
            out.add(dist.lower().replace('_', '-'))
    return out


def _modules_imported() -> Dict[str, List[str]]:
    """
    Walks every module under ``roicat/`` and collects the third-party top-level
    modules it imports, whether at module scope or inside a function.

    Standard-library modules and relative imports are dropped.

    Returns:
        (Dict[str, List[str]]):
            ``{module_name: ["path:lineno", ...]}``.
    """
    out = {}
    for path in sorted(DIR_PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                ## A non-zero level is a relative import: `from . import x`.
                names = [] if (node.level or node.module is None) else [node.module]
            else:
                continue
            for name in names:
                name_top = name.split('.')[0]
                if name_top in sys.stdlib_module_names or name_top == 'roicat':
                    continue
                site = f'{path.relative_to(PATH_ROOT)}:{node.lineno}'
                out.setdefault(name_top, []).append(site)
    return out


def test_every_import_is_declared_or_explicitly_optional():
    """
    Every third-party module ROICaT imports must either be in the ``core``
    extra or be listed in ``MODULES_NOT_DEPENDENCIES`` with a reason.

    This is the invariant that makes ``pip install roicat[core]`` mean
    something. It used to hold only by luck in places: ``pandas`` was undeclared
    for a long time and worked because seaborn pulled it in, and ``cv2`` was
    imported at module scope while ``core`` did not name it, so a
    ``roicat[classification]`` install could not ``import roicat`` at all
    (#662).
    """
    declared = _resolve(_load_pyproject()['project']['optional-dependencies'], 'core')
    undeclared = {}
    for module, sites in _modules_imported().items():
        if module in MODULES_NOT_DEPENDENCIES:
            continue
        dist = DISTS_BY_MODULE.get(module, module.lower().replace('_', '-'))
        if dist not in declared:
            undeclared[module] = sites[:3]
    assert not undeclared, (
        'Imported by ROICaT but not in the "core" extra: '
        f'{undeclared}. Either declare it, or add it to MODULES_NOT_DEPENDENCIES '
        'with a note saying why an install without it still works.'
    )


def test_optional_modules_are_not_imported_at_module_scope():
    """
    A module in ``MODULES_NOT_DEPENDENCIES`` is only optional if importing it is
    deferred to the moment it is used. At module scope it is a hard requirement
    wearing the wrong label, and ``import roicat`` fails without it.
    """
    offenders = {}
    for path in sorted(DIR_PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = (
                [alias.name for alias in node.names] if isinstance(node, ast.Import)
                else ([] if node.level or node.module is None else [node.module])
            )
            for name in names:
                if name.split('.')[0] in MODULES_NOT_DEPENDENCIES:
                    offenders[name] = f'{path.relative_to(PATH_ROOT)}:{node.lineno}'
    assert not offenders, (
        f'Optional modules imported at module scope: {offenders}. '
        'Move the import inside the function that uses it.'
    )


@pytest.mark.parametrize('extra_pinned, extra_latest', sorted(PAIRS_EXTRAS.items()))
def test_the_two_families_name_the_same_packages(extra_pinned, extra_latest):
    """
    ``latest`` exists so that a downstream package can install what ROICaT needs
    without inheriting its pins. A package present in one family and not the
    other makes it something else: either an unpinned hole in a set that claims
    to be exact, or an ImportError in somebody else's install.
    """
    extras = _load_pyproject()['project']['optional-dependencies']
    for name in (extra_pinned, extra_latest):
        assert name in extras, f"pyproject.toml has no '{name}' extra."

    pkgs_pinned, pkgs_latest = _resolve(extras, extra_pinned), _resolve(extras, extra_latest)
    assert pkgs_pinned == pkgs_latest, (
        f"'{extra_pinned}' and '{extra_latest}' have drifted apart. "
        f"Only in '{extra_pinned}': {sorted(pkgs_pinned - pkgs_latest)}. "
        f"Only in '{extra_latest}': {sorted(pkgs_latest - pkgs_pinned)}."
    )


def test_latest_extras_are_unpinned():
    """
    A version constraint inside ``latest`` defeats its purpose: it is the
    constraint, not the package, that a downstream package cannot satisfy.
    """
    extras = _load_pyproject()['project']['optional-dependencies']
    constrained = {}
    for name in PAIRS_EXTRAS.values():
        for req in extras[name]:
            body = req.split(';')[0].strip()
            ## Strip the distribution name and any [extras] before looking for a
            ## constraint, so 'holoviews[recommended]' does not read as one.
            rest = re.sub(r'^[A-Za-z0-9_.\-]+(\[[^\]]*\])?', '', body).strip()
            if rest and not body.lower().startswith('roicat['):
                constrained.setdefault(name, []).append(req)
    assert not constrained, f'Version constraints found in unpinned extras: {constrained}'


def test_base_dependencies_stay_empty():
    """
    The packages live in extras rather than in ``[project] dependencies``, and
    that is load-bearing rather than incidental.

    pip can add a constraint but never relax one. A pin in the base list would
    be inherited by every package depending on ROICaT with no way out, and a
    ``latest`` extra naming the same package unconstrained does not undo it --
    the resolver intersects the two and the pin wins. Verified directly: a
    package with ``packaging==24.0`` in its base dependencies and an extra
    asking for bare ``packaging`` still installs 24.0 when that extra is
    requested, with 26.3 available.

    So a pinned default and an unpinned opt-out cannot coexist if the packages
    are in the base list. Putting even one there starts undoing that.
    """
    declared = _load_pyproject()['project']['dependencies']
    assert declared == [], (
        f'[project] dependencies is no longer empty: {declared}. Anything here '
        'is pinned for every downstream consumer with no escape, and the '
        '"latest" extra cannot loosen it. Put it in "core" and "latest" instead.'
    )


@pytest.mark.parametrize('name', NAMES_EXTRAS_LEGACY)
def test_legacy_extras_still_resolve(name):
    """
    ``pip install roicat[all]`` is the documented install command, and the other
    names are printed in the README of every past release, in the docs and in an
    unknown number of downstream scripts. Removing one would make those commands
    warn and install nothing at all.
    """
    extras = _load_pyproject()['project']['optional-dependencies']
    assert name in extras, (
        f"pyproject.toml no longer declares the '{name}' extra. It is an alias "
        f"now, but `pip install roicat[{name}]` must keep working."
    )
    assert _resolve(extras, name), f"The '{name}' alias resolves to nothing."


######################################################################################################################################
######################################################## PYTHON VERSIONS #############################################################
######################################################################################################################################

PATH_BUILD_YML = PATH_ROOT / '.github' / 'workflows' / 'build.yml'

## Candidate minor versions to ask `requires-python` about. Wide enough that the
## range does not need revisiting when Python moves on.
VERSIONS_CANDIDATE = [f'3.{n}' for n in range(8, 30)]


def _versions_supported():
    """
    Returns:
        (set): the ``3.x`` strings that ``requires-python`` admits.
    """
    from packaging.specifiers import SpecifierSet

    spec = SpecifierSet(_load_pyproject()['project']['requires-python'])
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


def test_ci_covers_both_dependency_families():
    """
    The two families resolve to different versions and fail independently:
    ``core`` breaks when a pin stops being installable, ``latest`` breaks when a
    new upstream release is incompatible. The matrix has to build both or one of
    them is untested.
    """
    import yaml

    if not PATH_BUILD_YML.exists():
        pytest.skip(f'build.yml not found at {PATH_BUILD_YML}; not a source checkout.')
    with open(PATH_BUILD_YML, 'r') as f:
        workflow = yaml.safe_load(f)
    extras_tested = {str(v) for v in workflow['jobs']['build']['strategy']['matrix']['extra']}

    names_installed = {name for entry in extras_tested for name in entry.split(',')}
    assert 'dev' in names_installed, 'No CI row installs the pinned "core" family.'
    assert 'dev_latest' in names_installed, 'No CI row installs the unpinned "latest" family.'


def test_readme_states_the_supported_pythons():
    """
    The README's Requirements line is what users actually read, so it must name
    every supported version and nothing else.
    """
    path_readme = PATH_ROOT / 'README.md'
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
