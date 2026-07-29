"""Qt scheduling adapter for verified local file-source snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.file_sources import (
    ImageReader,
    SourceFileSnapshot,
    load_frozen_file_source_snapshot,
)
from napari_vipp.core.source_identity import LocalSourceIdentity


@dataclass(frozen=True, slots=True)
class SourceFileLoadSpec:
    """One path and series requested by an Image Source graph node."""

    node_id: str
    path: str
    series_index: int
    cache_key: tuple[object, ...]
    expected_identity: LocalSourceIdentity | None = None


@dataclass(frozen=True, slots=True)
class SourceFileLoadResult:
    """Snapshots and branch-local errors from one UI load attempt."""

    run_id: int
    snapshots: dict[tuple[object, ...], SourceFileSnapshot]
    error: str = ""
    node_id: str = ""
    errors: dict[tuple[object, ...], str] | None = None


class SourceFileLoadSignals(QObject):
    finished = Signal(object)


class SourceFileLoadWorker(QRunnable):
    """Materialize a group of source series away from Qt's GUI thread."""

    def __init__(
        self,
        run_id: int,
        specs: tuple[SourceFileLoadSpec, ...],
        *,
        reader: ImageReader,
    ) -> None:
        super().__init__()
        self.run_id = int(run_id)
        self.specs = tuple(specs)
        self.reader = reader
        self.signals = SourceFileLoadSignals()

    def run(self) -> None:
        snapshots: dict[tuple[object, ...], SourceFileSnapshot] = {}
        errors: dict[tuple[object, ...], str] = {}
        loaded_identities: dict[str, LocalSourceIdentity] = {}
        for spec in self.specs:
            try:
                expected_identity = spec.expected_identity or loaded_identities.get(
                    spec.path
                )
                snapshot = load_frozen_file_source_snapshot(
                    spec.path,
                    spec.series_index,
                    expected_identity=expected_identity,
                    reader=self.reader,
                )
                snapshots[spec.cache_key] = snapshot
                loaded_identities[spec.path] = snapshot.identity
            except Exception as exc:
                errors[spec.cache_key] = str(exc)
        self.signals.finished.emit(
            SourceFileLoadResult(self.run_id, snapshots, errors=errors)
        )


__all__ = [
    "SourceFileLoadResult",
    "SourceFileLoadSignals",
    "SourceFileLoadSpec",
    "SourceFileLoadWorker",
]
