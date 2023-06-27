__all__ = [
    'classifier',
]

for pkg in __all__:
    exec('from . import ' + pkg)