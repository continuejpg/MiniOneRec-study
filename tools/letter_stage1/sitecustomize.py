"""
process-level import guard (Stage 3).

Why this exists
---------------
Installing `k-means-constrained` during LETTER Stage 2 upgraded protobuf from the
3.x line to 6.33.6. `tensorboard 2.11.2` -- pulled in by `wandb` -- ships
generated *_pb2.py modules that require protobuf < 4, so merely importing wandb
now raises:

    TypeError: Descriptors cannot be created directly.
    If this call came from a _pb2.py file, your generated code is out of date...

`trl`/`transformers` import wandb unconditionally. Every formal run sets
WANDB_MODE=disabled, so wandb is never actually used -- it only has to import.

This module is loaded automatically by the interpreter (sitecustomize.py on
sys.path via PYTHONPATH) and makes `import wandb` resolve to an inert stub. It is
equivalent to uninstalling wandb for this process, but reversible and requiring
no source change and no package mutation.
"""
import importlib.abc
import importlib.machinery
import sys
import types

BLOCKED = ("wandb",)


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(
                f"sitecustomize: import of {fullname!r} blocked "
                f"(WANDB_MODE=disabled; tensorboard/protobuf version conflict)",
                name=fullname)
        return None


# install only once
if not any(getattr(f, "_dsh_import_guard", False) for f in sys.meta_path):
    _f = _Blocker()
    _f._dsh_import_guard = True
    sys.meta_path.insert(0, _f)
