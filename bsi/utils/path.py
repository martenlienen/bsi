from os import PathLike
from pathlib import Path


def relative_to_project_root(path: PathLike | None = None) -> Path:
    root = Path(__file__).parent.parent.parent
    if path is None:
        return root
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return root / path
