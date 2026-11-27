"""Cross-engine benchmark harness for audiosronnx.

This package scores every registered engine on a fixed, externally-hosted gold
set with the ``speechonnxmetrics`` library, then renders the results into the
docs. It is intentionally kept out of the runtime package: it lives behind the
``benchmark`` install extra and is never imported by ``audiosronnx`` itself.

The entry point is the CLI, ``python -m benchmarks.run`` (see :mod:`benchmarks.run`).
"""
