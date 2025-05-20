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
    exec('from . import ' + pkg)

from .__main__ import run_pipeline

__version__ = '1.5.0'
