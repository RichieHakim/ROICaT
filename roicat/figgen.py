from pathlib import Path
import matplotlib.pyplot as plt

def save_fig_types(fig, base_folder, fn, types=['png', 'svg', 'pdf', 'eps']):
    """
    Saves figures as multiple different types.
    fig: matplotlib.pyplot.figure
        matplotlib figure to save
    base_fn: str
        base_filename (including directory) to which to append extension
    types:
        list of file extensions to include as figure saves
    """
    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)
    for typ in types:
        base_file = str((base_path / f'{fn}.{typ}').resolve())
        fig.savefig(base_file)
    return
