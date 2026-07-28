"""Optional native Imaris IMS writer."""

from __future__ import annotations

import platform
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp.core.channel_colors import FLUORESCENCE_COLORS, color_value_to_rgb
from napari_vipp.core.metadata import AxisMetadata, ImageState

SUPPORTED_IMARIS_DTYPES = frozenset(("uint8", "uint16", "uint32", "float32"))
_CANONICAL_AXES = ("t", "c", "z", "y", "x")
_MAX_BLOCK_SIZE = {"t": 1, "c": 1, "z": 16, "y": 256, "x": 256}


class OptionalImarisWriterError(ImportError):
    """Raised when direct IMS export is unavailable."""


def write_imaris(
    data,
    path: str | Path,
    *,
    image_state: ImageState | dict[str, Any] | None = None,
) -> Path:
    """Write one image as an Imaris 5.5 ``.ims`` dataset."""
    output_path = Path(path)
    state = _coerce_state(image_state)
    shape = tuple(int(size) for size in getattr(data, "shape", ()))
    if not shape:
        raise ValueError("IMS export requires a non-scalar image array.")
    if any(size <= 0 for size in shape):
        raise ValueError("IMS export requires a non-empty image array.")
    dtype_value = getattr(data, "dtype", None)
    dtype = np.dtype(dtype_value if dtype_value is not None else np.asarray(data).dtype)
    if dtype.name not in SUPPORTED_IMARIS_DTYPES:
        supported = ", ".join(sorted(SUPPORTED_IMARIS_DTYPES))
        raise ValueError(
            f"IMS export does not silently convert {dtype.name}. Convert the "
            f"image explicitly to one of: {supported}."
        )
    source_axes, sizes = _resolve_axes(shape, state)
    _validate_spatial_units(state)
    writer = _load_writer()
    try:
        _write_with_imariswriter(
            writer,
            data,
            output_path,
            dtype,
            source_axes,
            sizes,
            state,
        )
    except Exception as exc:
        detail = f" {exc}" if str(exc).strip() else ""
        raise RuntimeError(
            f"ImarisWriter failed to write {output_path}.{detail}"
        ) from exc
    return output_path


def _load_writer():
    try:
        package = import_module("napari_vipp_imaris")
    except ImportError as exc:
        system = platform.system()
        machine = platform.machine()
        availability = (
            "The native writer currently supports macOS arm64 only. "
            if (system, machine) != ("Darwin", "arm64")
            else ""
        )
        raise OptionalImarisWriterError(
            "IMS export requires the optional native writer. "
            f'{availability}Install it with: pip install "napari-vipp[ims]"'
        ) from exc
    writer = getattr(package, "PyImarisWriter", None)
    if writer is None:
        raise OptionalImarisWriterError(
            "The installed napari-vipp-imaris package is invalid: "
            "PyImarisWriter is missing."
        )
    return writer


def _resolve_axes(
    shape: tuple[int, ...],
    state: ImageState | None,
) -> tuple[tuple[str | None, ...], dict[str, int]]:
    if state is None or len(state.axes) != len(shape):
        raise ValueError(
            "IMS export requires carried axis metadata matching the array. "
            "Load the image through VIPP or define its axes before export."
        )
    source_axes: list[str | None] = []
    sizes = dict.fromkeys(_CANONICAL_AXES, 1)
    for axis, size in zip(state.axes, shape, strict=True):
        key = _imaris_axis_key(axis)
        if key is None:
            if size != 1:
                raise ValueError(
                    f"IMS export cannot map non-singleton axis {axis.name!r}. "
                    "Only X, Y, Z, C/RGB(A), and T are supported."
                )
            source_axes.append(None)
            continue
        if size != 1 and not axis.is_explicit:
            raise ValueError(
                f"IMS export requires explicit semantics for axis {axis.name!r}."
            )
        if key in source_axes:
            raise ValueError(f"IMS export received duplicate {key.upper()} axes.")
        source_axes.append(key)
        sizes[key] = size
    return tuple(source_axes), {key: int(value) for key, value in sizes.items()}


def _imaris_axis_key(axis: AxisMetadata) -> str | None:
    name = axis.name.strip().casefold()
    if name in {"x", "y", "z", "t"}:
        return name
    if name in {"c", "channel", "rgb", "rgba", "s"}:
        return "c"
    if axis.type == "channel":
        return "c"
    if axis.type == "time":
        return "t"
    return None


def _write_with_imariswriter(
    writer,
    data,
    path: Path,
    dtype: np.dtype,
    source_axes: tuple[str | None, ...],
    sizes: dict[str, int],
    state: ImageState,
) -> None:
    image_size = writer.ImageSize(**sizes)
    block_sizes = {
        key: min(sizes[key], _MAX_BLOCK_SIZE[key]) for key in _CANONICAL_AXES
    }
    block_size = writer.ImageSize(**block_sizes)
    sample_size = writer.ImageSize(**dict.fromkeys(_CANONICAL_AXES, 1))
    sequence = writer.DimensionSequence("x", "y", "z", "c", "t")
    options = writer.Options()

    class SilentCallback(writer.CallbackClass):
        def RecordProgress(self, progress: float, total_bytes_written: int):
            return None

    converter = None
    try:
        converter = writer.ImageConverter(
            dtype.name,
            image_size,
            sample_size,
            sequence,
            block_size,
            str(path),
            options,
            "napari-vipp",
            VIPP_VERSION,
            SilentCallback(),
        )
        for starts in _block_starts(sizes, block_sizes):
            block_index = writer.ImageSize(
                **{key: starts[key] // block_sizes[key] for key in _CANONICAL_AXES}
            )
            if converter.NeedCopyBlock(block_index):
                converter.CopyBlock(
                    _extract_block(
                        data,
                        source_axes,
                        starts,
                        sizes,
                        block_sizes,
                        dtype,
                    ),
                    block_index,
                )
        converter.Finish(
            _image_extents(writer, state, sizes),
            _parameters(writer, state),
            _time_infos(state, sizes["t"]),
            _color_infos(writer, state, sizes["c"]),
            True,
        )
    finally:
        if converter is not None:
            converter.Destroy()


def _block_starts(sizes: dict[str, int], blocks: dict[str, int]):
    for t in range(0, sizes["t"], blocks["t"]):
        for c in range(0, sizes["c"], blocks["c"]):
            for z in range(0, sizes["z"], blocks["z"]):
                for y in range(0, sizes["y"], blocks["y"]):
                    for x in range(0, sizes["x"], blocks["x"]):
                        yield {"t": t, "c": c, "z": z, "y": y, "x": x}


def _extract_block(
    data,
    source_axes: tuple[str | None, ...],
    starts: dict[str, int],
    sizes: dict[str, int],
    blocks: dict[str, int],
    dtype: np.dtype,
) -> np.ndarray:
    slices = []
    for key in source_axes:
        if key is None:
            slices.append(slice(0, 1))
        else:
            stop = min(starts[key] + blocks[key], sizes[key])
            slices.append(slice(starts[key], stop))
    chunk = data[tuple(slices)]
    compute = getattr(chunk, "compute", None)
    if callable(compute):
        chunk = compute()
    chunk = np.asarray(chunk, dtype=dtype)
    ignored = tuple(index for index, key in enumerate(source_axes) if key is None)
    if ignored:
        chunk = np.squeeze(chunk, axis=ignored)
    present = [key for key in source_axes if key is not None]
    ordered = [key for key in _CANONICAL_AXES if key in present]
    if len(present) > 1:
        chunk = np.transpose(chunk, tuple(present.index(key) for key in ordered))
    valid_shape = tuple(
        min(blocks[key], sizes[key] - starts[key]) if key in present else 1
        for key in _CANONICAL_AXES
    )
    chunk = chunk.reshape(valid_shape)
    padded = np.zeros(tuple(blocks[key] for key in _CANONICAL_AXES), dtype=dtype)
    padded[tuple(slice(0, size) for size in valid_shape)] = chunk
    return np.ascontiguousarray(padded)


def _image_extents(writer, state: ImageState, sizes: dict[str, int]):
    extents = {}
    for key in ("x", "y", "z"):
        axis = _metadata_axis(state, key)
        if axis is None:
            extents[key] = (0.0, float(sizes[key]))
            continue
        factor = _micrometer_factor(axis.unit)
        start = float(axis.translation) * factor
        extents[key] = (start, start + float(axis.scale) * factor * sizes[key])
    return writer.ImageExtents(
        extents["x"][0],
        extents["y"][0],
        extents["z"][0],
        extents["x"][1],
        extents["y"][1],
        extents["z"][1],
    )


def _micrometer_factor(unit: str | None) -> float:
    text = str(unit or "").strip().casefold().replace("μ", "µ")
    factors = {
        "": 1.0,
        "pixel": 1.0,
        "pixels": 1.0,
        "px": 1.0,
        "m": 1_000_000.0,
        "meter": 1_000_000.0,
        "meters": 1_000_000.0,
        "metre": 1_000_000.0,
        "metres": 1_000_000.0,
        "mm": 1_000.0,
        "millimeter": 1_000.0,
        "millimeters": 1_000.0,
        "millimetre": 1_000.0,
        "millimetres": 1_000.0,
        "nm": 0.001,
        "nanometer": 0.001,
        "nanometers": 0.001,
        "nanometre": 0.001,
        "nanometres": 0.001,
        "um": 1.0,
        "µm": 1.0,
        "micrometer": 1.0,
        "micrometers": 1.0,
        "micrometre": 1.0,
        "micrometres": 1.0,
    }
    if text not in factors:
        raise ValueError(
            f"IMS export cannot convert spatial unit {unit!r} to micrometers."
        )
    return factors[text]


def _validate_spatial_units(state: ImageState) -> None:
    for key in ("x", "y", "z"):
        axis = _metadata_axis(state, key)
        if axis is not None:
            _micrometer_factor(axis.unit)


def _metadata_axis(state: ImageState, key: str) -> AxisMetadata | None:
    for axis in state.axes:
        if _imaris_axis_key(axis) == key:
            return axis
    return None


def _parameters(writer, state: ImageState):
    parameters = writer.Parameters()
    if state.source_name:
        parameters.set_value("Image", "Info", state.source_name)
    parameters.set_value("VIPP", "Version", VIPP_VERSION)
    parameters.set_value("VIPP", "AxisOrder", state.axis_order)
    if state.history:
        parameters.set_value("VIPP", "History", " | ".join(state.history))
    if state.source.uri:
        parameters.set_value("VIPP", "SourceURI", state.source.uri)
    if state.source.format:
        parameters.set_value("VIPP", "SourceFormat", state.source.format)
    for index, channel in enumerate(state.channels):
        if channel.name:
            parameters.set_channel_name(index, channel.name)
    return parameters


def _time_infos(state: ImageState, count: int) -> list[datetime]:
    base = _parse_datetime(state.acquisition.acquisition_date) or datetime.now(UTC)
    axis = _metadata_axis(state, "t")
    step = _seconds_per_step(axis) if axis is not None else 0.0
    return [
        (base + timedelta(seconds=step * index)).replace(tzinfo=None)
        for index in range(count)
    ]


def _parse_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds_per_step(axis: AxisMetadata) -> float:
    unit = str(axis.unit or "").strip().casefold()
    factors = {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 0.001,
        "millisecond": 0.001,
        "milliseconds": 0.001,
        "min": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hour": 3600.0,
        "hours": 3600.0,
    }
    return float(axis.scale) * factors.get(unit, 0.0)


def _color_infos(writer, state: ImageState, count: int):
    infos = []
    for index in range(count):
        info = writer.ColorInfo()
        channel = state.channels[index] if index < len(state.channels) else None
        rgb = color_value_to_rgb(channel.color if channel is not None else None)
        if rgb is None:
            rgb = FLUORESCENCE_COLORS[index % len(FLUORESCENCE_COLORS)]
        info.set_base_color(
            writer.Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        )
        infos.append(info)
    return infos


def _coerce_state(value: ImageState | dict[str, Any] | None) -> ImageState | None:
    if isinstance(value, ImageState):
        return value
    if isinstance(value, dict):
        return ImageState.from_dict(value)
    return None


__all__ = [
    "OptionalImarisWriterError",
    "SUPPORTED_IMARIS_DTYPES",
    "write_imaris",
]
