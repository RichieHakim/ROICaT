__version__ = '1.7.2'

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
