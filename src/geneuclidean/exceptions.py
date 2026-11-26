"""Structured, backward-compatible exception hierarchy for GenEuclidean.

Design principle (Google Python API guidelines):
    Callers should be able to catch domain-specific library failures via
    ``except ge.GenEuclideanError`` while existing code that catches standard
    built-in exceptions (``ValueError``, ``KeyError``, ``FileNotFoundError``,
    ``RuntimeError``) continues to work without any breaking changes.

    Every concrete exception therefore uses cooperative dual inheritance
    ``(GenEuclideanError, <StdlibError>)`` and supports an optional ``hint``
    string that surfaces actionable remediation guidance (e.g., the exact
    ``pip install 'geneuclidean[...]'`` extra to install).
"""

from __future__ import annotations


class GenEuclideanError(Exception):
    """Base class for all GenEuclidean library errors.

    Attributes:
        message: Human-readable error summary.
        hint: Optional actionable suggestion appended to the formatted message.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        full = f"{message} Hint: {hint}" if hint else message
        super().__init__(full)

    def __str__(self) -> str:
        # Override to prevent KeyError.__str__ from double-quoting error messages
        # in subclasses that inherit from both GenEuclideanError and KeyError.
        if self.hint:
            return f"{self.message} Hint: {self.hint}"
        return self.message


class RegistryError(GenEuclideanError, KeyError):
    """Raised when plugin/component registration or lookup fails."""


class TargetNotFoundError(GenEuclideanError, KeyError):
    """Raised when a CNS or custom protein target identifier cannot be resolved."""


class InvalidMoleculeError(GenEuclideanError, ValueError):
    """Raised when a SMILES, SAFE, SELFIES, or RDKit Mol input is invalid."""


class PocketFeaturizationError(GenEuclideanError, ValueError):
    """Raised when 3D binding-pocket extraction or featurization fails."""


class StructureNotFoundError(GenEuclideanError, FileNotFoundError):
    """Raised when a required 3D structure (PDB/mmCIF/SDF) or artifact file is missing."""


class CheckpointError(GenEuclideanError, ValueError):
    """Raised when loading or validating a model/adapter checkpoint fails."""


class OptionalDependencyError(GenEuclideanError, RuntimeError):
    """Raised when an optional dependency extra or external binary is unavailable."""


class DockingError(GenEuclideanError, RuntimeError):
    """Raised when 3D conformer generation or AutoDock Vina execution fails."""


__all__ = [
    "CheckpointError",
    "DockingError",
    "GenEuclideanError",
    "InvalidMoleculeError",
    "OptionalDependencyError",
    "PocketFeaturizationError",
    "RegistryError",
    "StructureNotFoundError",
    "TargetNotFoundError",
]
