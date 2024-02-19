import re
import os

def increment_version(path: str):
    """
    Increment the version number in the file at the given path
    RH 2024

    Args:
        path (str):
            Path to the file containing the __version__ variable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, 'r') as file:
        content = file.readlines()

    for i, line in enumerate(content):
        if line.startswith('__version__'):
            version_match = re.search(r"'([^']*)'", line)
            if version_match:
                current_version = version_match.group(1)
                version_parts = current_version.split('.')
                version_parts[-1] = str(int(version_parts[-1]) + 1)
                new_version = '.'.join(version_parts)
                content[i] = f"__version__ = '{new_version}'\n"
                break

    with open(path, 'w') as file:
        file.writelines(content)

    print(f"Version updated to {new_version}")

import argparse
parser = argparse.ArgumentParser(description='Increment version number in a file')
## Ingest --path argument
parser.add_argument('--path', type=str, help='Path to the version file', required=True)
args = parser.parse_args()
path_version_file = args.path

if __name__ == "__main__":
    increment_version(path_version_file)
