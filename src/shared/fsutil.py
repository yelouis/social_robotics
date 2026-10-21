from pathlib import Path
from typing import List, Union


def is_appledouble(path: Union[Path, str]) -> bool:
    """Return True if the filename starts with '._' (macOS AppleDouble twin file)."""
    return Path(path).name.startswith("._")


def data_glob(
    directory: Union[Path, str],
    pattern: str,
    recursive: bool = False,
) -> List[Path]:
    """Sorted glob or rglob matches over directory, excluding AppleDouble files."""
    d = Path(directory)
    matches = d.rglob(pattern) if recursive else d.glob(pattern)
    return sorted(p for p in matches if not is_appledouble(p))
