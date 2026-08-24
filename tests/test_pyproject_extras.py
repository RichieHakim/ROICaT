"""
Checks on the dependency declaration in ``pyproject.toml``.

Everything ROICaT imports is declared in ``[project] dependencies``, so a plain
``pip install roicat`` is a working install. Two things can quietly break that:
a new ``import`` of something nobody declared (it works locally because some
other package happened to pull it in), and the ``pinned`` extra drifting away
from the base list. Both are tested here.

The base list carries lower bounds only. A package that depends on ROICaT
inherits the base requirements and has no way to relax them, so an upper bound
here constrains every downstream consumer. Exact versions belong in ``pinned``.
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

## Extras that predate the move to a self-sufficient base install. They are
## empty now, but removing them would break `pip install roicat[all]` and the
## other commands printed in older docs, papers and downstream scripts.
NAMES_EXTRAS_LEGACY = [
    'core',
    'classification',
    'tracking',
    'all',
    'core_latest',
    'classification_latest',
    'tracking_latest',
    'all_latest',
    'dev_latest',
]

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
    Every third-party module ROICaT imports must either be a base dependency or
    be listed in ``MODULES_NOT_DEPENDENCIES`` with a reason.

    This is the invariant that makes ``pip install roicat`` mean something. It
    used to hold only by luck in places: ``pandas`` was undeclared for a long
    time and worked because seaborn pulled it in, and ``cv2`` was imported at
    module scope while living in an extra, so a ``roicat[classification]``
    install could not ``import roicat`` at all (#662).
    """
    declared = _names_declared(_load_pyproject()['project']['dependencies'])
    undeclared = {}
    for module, sites in _modules_imported().items():
        if module in MODULES_NOT_DEPENDENCIES:
            continue
        dist = DISTS_BY_MODULE.get(module, module.lower().replace('_', '-'))
        if dist not in declared:
            undeclared[module] = sites[:3]
    assert not undeclared, (
        'Imported by ROICaT but not in [project] dependencies: '
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


def test_pinned_names_the_same_packages_as_the_base_dependencies():
    """
    ``pinned`` exists to reproduce a known-good install of the base list. A
    package present in one and not the other makes it something else: either an
    unpinned hole in a set that claims to be exact, or a package that arrives
    only when you ask for the pins.
    """
    project = _load_pyproject()['project']
    base = _names_declared(project['dependencies'])
    pinned = _names_declared([
        req for req in project['optional-dependencies']['pinned']
        if not req.lower().startswith('roicat[')
    ])
    assert base == pinned, (
        f'Base dependencies and the "pinned" extra have drifted apart. '
        f'Only in base: {sorted(base - pinned)}. Only in pinned: {sorted(pinned - base)}.'
    )


def test_base_dependencies_carry_no_upper_bounds():
    """
    An upper bound in the base list cannot be relaxed by anything that depends
    on ROICaT, so it silently caps every downstream package too. Upper bounds
    belong in ``pinned``, which is opt-in.
    """
    bounded = []
    for req in _load_pyproject()['project']['dependencies']:
        body = req.split(';')[0].strip()
        ## Strip the name and any [extras] first, so `holoviews[recommended]`
        ## does not read as a constraint.
        constraint = re.sub(r'^[A-Za-z0-9_.\-]+(\[[^\]]*\])?', '', body).strip()
        if any(op in constraint for op in ('<', '==', '~=')):
            bounded.append(req)
    assert not bounded, (
        f'Upper-bounded or exactly-pinned base dependencies: {bounded}. '
        'Move the constraint to the "pinned" extra.'
    )


@pytest.mark.parametrize('name', NAMES_EXTRAS_LEGACY)
def test_legacy_extras_still_resolve(name):
    """
    ``pip install roicat[all]`` is printed in the README of every past release,
    in the docs and in an unknown number of downstream scripts. Removing the
    name would make those commands warn and install a different thing than the
    author intended, so the names stay even though they are empty.
    """
    extras = _load_pyproject()['project']['optional-dependencies']
    assert name in extras, (
        f"pyproject.toml no longer declares the '{name}' extra. It is empty by "
        f"design, but `pip install roicat[{name}]` must keep working."
    )


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


def test_ci_covers_both_the_base_and_pinned_dependency_sets():
    """
    The two dependency sets resolve to different versions and can fail
    independently: ``pinned`` breaks when a pin stops being installable, the
    base list breaks when a new upstream release is incompatible. The matrix
    has to build both or one of them is untested.
    """
    import yaml

    if not PATH_BUILD_YML.exists():
        pytest.skip(f'build.yml not found at {PATH_BUILD_YML}; not a source checkout.')
    with open(PATH_BUILD_YML, 'r') as f:
        workflow = yaml.safe_load(f)
    extras_tested = {str(v) for v in workflow['jobs']['build']['strategy']['matrix']['extra']}

    names_installed = {name for entry in extras_tested for name in entry.split(',')}
    assert 'dev' in names_installed, 'No CI row installs the unpinned base dependencies.'
    assert 'dev_pinned' in names_installed, 'No CI row installs the "pinned" set.'


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
