# Pre-import torch so it is present in sys.modules before any patch.dict context.
# This prevents torch C-extension re-initialization errors when tests use
# patch.dict("sys.modules", ...) and cause module cache eviction.
import torch  # noqa: F401
