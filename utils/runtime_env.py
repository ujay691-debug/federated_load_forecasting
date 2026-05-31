import os
import sys


_DLL_DIR_HANDLES = []


def ensure_conda_dll_paths():
    if os.name != "nt":
        return

    conda_prefix = sys.prefix or os.environ.get("CONDA_PREFIX")
    candidate_dirs = [
        conda_prefix,
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
        os.path.join(conda_prefix, "Library", "usr", "bin"),
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Scripts"),
        os.path.join(conda_prefix, "DLLs"),
        os.path.join(conda_prefix, "bin"),
    ]

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in candidate_dirs:
        if not os.path.isdir(dll_dir):
            continue
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir))
            except OSError:
                pass
        if dll_dir not in path_parts:
            path_parts.insert(0, dll_dir)

    os.environ["PATH"] = os.pathsep.join(path_parts)
