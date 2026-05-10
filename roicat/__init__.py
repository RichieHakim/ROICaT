import warnings
__version__ = '1.7.3'

__all__ = [
    'classification',
    'tracking',
    'model_training',
    'data_importing',
    'helpers',
    'pipelines',
    'ROInet',
    'util',
    'visualization',
]

for pkg in __all__:
    try:
        exec('from . import ' + pkg)
    except ImportError as e:
        warnings.warn(f"roicat.{pkg} unavailable. Install the relevant extra if you need to use it.")

from .__main__ import run_pipeline
