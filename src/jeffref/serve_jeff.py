"""jeff-serve (Jeff's own server, unchanged) with MLX's buffer cache capped.

MLX keeps freed GPU buffers in a cache for reuse, with no limit by default. With prompts of many different lengths the
cache grows to many gigabytes (14 GB measured after a few demo tickets) although the model itself needs about 1.5 GB.
JEFF_MLX_CACHE_LIMIT_GB caps it (set in scripts/serve_jeff.sh)."""

import os

import mlx.core as mx


def main() -> None:
    value = os.environ.get("JEFF_MLX_CACHE_LIMIT_GB")
    if value is None:
        raise RuntimeError("Set JEFF_MLX_CACHE_LIMIT_GB (for example 1) to cap MLX's buffer cache")
    mx.set_cache_limit(int(float(value) * 2**30))
    from jeff.server import main as serve

    serve()


if __name__ == "__main__":
    main()
