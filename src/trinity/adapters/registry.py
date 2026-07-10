"""The benchmark-adapter registry: one lookup for the whole evaluation pipeline.

The registry is a process-global name -> :class:`BenchmarkAdapter` map. The core
evaluator resolves a benchmark once via :func:`get_adapter` and never touches a
benchmark by name again, which is what keeps benchmark-specific logic behind the
adapter seam.

Registration is available two ways:

* as a decorator on an adapter *class* (instantiated with no arguments)::

      @register_adapter("math500")
      class Math500Adapter(BenchmarkAdapter): ...

* by handing over an already-built adapter *instance* (used by the built-ins,
  which share one parametrised adapter class)::

      register_adapter("math500", DelegatingBenchmarkAdapter("math500"))

Names are matched case-insensitively and stored normalised (stripped,
lower-cased) so ``"MATH500"`` and ``"math500"`` resolve to the same adapter.
"""
from __future__ import annotations

from typing import Callable, overload

from .base import BenchmarkAdapter

__all__ = [
    "register_adapter",
    "get_adapter",
    "is_registered",
    "available_adapters",
    "clear_registry",
]

# name (normalised) -> adapter instance
_REGISTRY: dict[str, BenchmarkAdapter] = {}


def _normalise(name: str) -> str:
    """Canonicalise a benchmark name for stable, case-insensitive lookup."""
    return (name or "").strip().lower()


@overload
def register_adapter(name: str) -> Callable[[type[BenchmarkAdapter]], type[BenchmarkAdapter]]: ...
@overload
def register_adapter(name: str, adapter: BenchmarkAdapter) -> BenchmarkAdapter: ...


def register_adapter(name, adapter=None):
    """Register ``adapter`` under ``name``, or return a class decorator.

    Called with an instance, registers it and returns it. Called with only a
    name, returns a decorator that registers the decorated
    :class:`BenchmarkAdapter` subclass (instantiated with no arguments) and
    returns the class unchanged.

    Args:
        name: The benchmark identifier to register under (case-insensitive).
        adapter: An adapter instance, or ``None`` to use decorator form.

    Returns:
        The registered instance, or a class decorator.

    Raises:
        ValueError: If ``name`` is empty or already registered.
        TypeError: If ``adapter`` is not a :class:`BenchmarkAdapter`.
    """
    key = _normalise(name)
    if not key:
        raise ValueError("Benchmark adapter name must be a non-empty string.")

    def _register(instance: BenchmarkAdapter) -> BenchmarkAdapter:
        if not isinstance(instance, BenchmarkAdapter):
            raise TypeError(
                f"Expected a BenchmarkAdapter, got {type(instance).__name__}."
            )
        if key in _REGISTRY:
            raise ValueError(f"Benchmark adapter {key!r} is already registered.")
        _REGISTRY[key] = instance
        return instance

    if adapter is not None:
        return _register(adapter)

    def _decorator(cls: type[BenchmarkAdapter]) -> type[BenchmarkAdapter]:
        _register(cls())
        return cls

    return _decorator


def get_adapter(name: str) -> BenchmarkAdapter:
    """Return the adapter registered for ``name``.

    Args:
        name: The benchmark identifier (case-insensitive).

    Returns:
        The registered :class:`BenchmarkAdapter`.

    Raises:
        KeyError: If no adapter is registered for ``name``; the message lists the
            available benchmarks so the caller can see valid options.
    """
    key = _normalise(name)
    try:
        return _REGISTRY[key]
    except KeyError:
        avail = ", ".join(available_adapters()) or "<none>"
        raise KeyError(
            f"No benchmark adapter registered for {name!r}. Available: {avail}."
        ) from None


def is_registered(name: str) -> bool:
    """Return whether an adapter is registered for ``name`` (case-insensitive)."""
    return _normalise(name) in _REGISTRY


def available_adapters() -> tuple[str, ...]:
    """Return the sorted tuple of registered benchmark names."""
    return tuple(sorted(_REGISTRY))


def clear_registry() -> None:
    """Remove all registrations. Intended for tests that register throwaway adapters."""
    _REGISTRY.clear()
