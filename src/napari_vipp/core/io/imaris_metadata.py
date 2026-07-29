"""Read or synthesize an Imaris ``/DataSetInfo`` parameter tree."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np


class ImarisMetadataError(RuntimeError):
    """Raised when an IMS source cannot be checkpointed without metadata loss."""


def read_imaris_dataset_info(path: str | Path) -> dict[str, dict[str, str]]:
    """Return every decoded Imaris ``/DataSetInfo`` parameter.

    Imaris stores ordinary parameter values as HDF5 attributes and long values
    as datasets. Both representations are normalized to the strings accepted by
    ImarisWriter.
    """
    source = Path(path)
    if not source.is_file():
        raise ImarisMetadataError(
            f"Cannot preserve Imaris metadata because the source is missing: {source}"
        )
    h5py = _load_h5py()
    try:
        with h5py.File(source, "r") as handle:
            if "DataSetInfo" not in handle:
                raise ImarisMetadataError(
                    f"Cannot preserve Imaris metadata: {source} has no /DataSetInfo."
                )
            root = handle["DataSetInfo"]
            sections: dict[str, dict[str, str]] = {}
            for encoded_section, group in root.items():
                if not isinstance(group, h5py.Group):
                    raise ImarisMetadataError(
                        "Cannot preserve Imaris metadata: /DataSetInfo contains "
                        f"an unsupported item {encoded_section!r}."
                    )
                section = _decode_name(encoded_section)
                parameters = {
                    str(name): _decode_value(value)
                    for name, value in group.attrs.items()
                }
                for encoded_name, item in group.items():
                    if not isinstance(item, h5py.Dataset):
                        raise ImarisMetadataError(
                            "Cannot preserve Imaris metadata: section "
                            f"{section!r} contains an unsupported nested group."
                        )
                    name = _decode_name(encoded_name)
                    value = _decode_value(item[()])
                    if name in parameters and parameters[name] != value:
                        raise ImarisMetadataError(
                            "Cannot preserve Imaris metadata: duplicate parameter "
                            f"{section}/{name}."
                        )
                    parameters[name] = value
                sections[section] = parameters
    except ImarisMetadataError:
        raise
    except Exception as exc:
        raise ImarisMetadataError(
            f"Cannot preserve Imaris /DataSetInfo metadata from {source}: {exc}"
        ) from exc
    if not sections:
        raise ImarisMetadataError(
            f"Cannot preserve Imaris metadata: {source} has an empty /DataSetInfo."
        )
    return sections


def read_lif_dataset_info(
    path: str | Path,
    series_index: int = 0,
    *,
    require_complete: bool = False,
) -> dict[str, dict[str, str]]:
    """Synthesize Imaris parameters from one Leica LIF image's XML metadata."""
    source = Path(path)
    if not source.is_file():
        raise ImarisMetadataError(
            f"Cannot preserve Leica metadata because the source is missing: {source}"
        )
    liffile = _load_liffile()
    try:
        with liffile.LifFile(str(source)) as lif:
            images = _container_values(getattr(lif, "images", None))
            if not 0 <= int(series_index) < len(images):
                raise ImarisMetadataError(
                    f"Cannot preserve Leica metadata: series {series_index} is not "
                    f"available in {source}."
                )
            image = images[int(series_index)]
            image_root = getattr(image, "xml_element", None)
            file_root = getattr(lif, "xml_element", None)
            if image_root is None or file_root is None:
                raise ImarisMetadataError(
                    "Cannot preserve Leica metadata: liffile did not expose the "
                    "selected image XML."
                )
            sections = _lif_parameter_sections(
                source,
                image,
                image_root,
                file_root,
                str(getattr(liffile, "__version__", "unknown")),
            )
            if require_complete:
                _require_complete_lif_metadata(source, image, sections)
    except ImarisMetadataError:
        raise
    except Exception as exc:
        raise ImarisMetadataError(
            f"Cannot preserve Leica metadata from {source}: {exc}"
        ) from exc
    return sections


def _lif_parameter_sections(
    source: Path,
    image: Any,
    image_root: ElementTree.Element,
    file_root: ElementTree.Element,
    liffile_version: str,
) -> dict[str, dict[str, str]]:
    channels = _xml_elements(image_root, "ChannelDescription")
    dimensions = _xml_elements(image_root, "DimensionDescription")
    setting = _first_xml_element(
        image_root,
        "ATLConfocalSettingDefinition",
        required_attribute="StagePosX",
    )
    if setting is None:
        setting = _first_xml_element(image_root, "ATLConfocalSettingDefinition")
    hardware = next(
        (
            element
            for element in _xml_elements(image_root, "Attachment")
            if element.attrib.get("Name") == "HardwareSetting"
        ),
        None,
    )
    experiment = _first_xml_element(file_root, "Experiment")
    header = _first_xml_element(image_root, "LMSDataContainerHeader")
    if header is None:
        header = _first_xml_element(file_root, "LMSDataContainerHeader")
    image_element = _first_xml_element(image_root, "Image")

    name = str(image_root.attrib.get("Name", "") or getattr(image, "name", ""))
    description = ""
    if image_element is not None:
        description = str(image_element.attrib.get("TextDescription", ""))
    recording_date = _lif_recording_date(image)
    setting_values = setting.attrib if setting is not None else {}
    hardware_values = hardware.attrib if hardware is not None else {}
    image_values = {
        "Description": description,
        "ElementName": name,
        "ExperimentPath": (
            str(experiment.attrib.get("Path", "")) if experiment is not None else ""
        ),
        "File Version": (
            str(header.attrib.get("Version", "")) if header is not None else ""
        ),
        "FileName": str(source),
        "LensPower": str(setting_values.get("Magnification", "")),
        "MicroscopeModality": str(hardware_values.get("SystemTypeName", "")),
        "MicroscopeModel": str(setting_values.get("MicroscopeModel", "")),
        "Name": name,
        "NumberOfChannels": str(len(channels)),
        "NumericalAperture": str(setting_values.get("NumericalAperture", "")),
        "ObjectiveImmersion": str(setting_values.get("Immersion", "")),
        "ObjectiveName": str(setting_values.get("ObjectiveName", "")).strip(),
        "OriginalFormat": "Leica: Image File Format LIF",
        "OriginalFormatFileIOVersion": f"liffile {liffile_version}",
        "RecordingDate": recording_date,
        "RefractionIndex": str(setting_values.get("RefractionIndex", "")),
        "ResampleDimensionX": "true",
        "ResampleDimensionY": "true",
        "ResampleDimensionZ": "true",
        "Unit": "um",
    }
    sections: dict[str, dict[str, str]] = {
        "Image": image_values,
        "Log": {"Entries": "0"},
    }

    dimension_by_axis: dict[str, dict[str, str]] = {}
    dimension_names = {"1": "X", "2": "Y", "3": "Z", "4": "T"}
    for index, element in enumerate(dimensions, start=1):
        values = {str(key): str(value) for key, value in element.attrib.items()}
        axis = dimension_names.get(values.get("DimID", ""), str(index))
        sections[f"Dimension {axis}"] = values
        if axis in {"X", "Y", "Z"}:
            dimension_by_axis[axis] = values
            image_values = sections["Image"]
            image_values[axis] = values.get("NumberOfElements", "")

    sections["Image"].update(_lif_extents(dimension_by_axis, setting_values))
    emissions = _lif_emission_wavelengths(setting)
    pinhole = _scaled_pinhole(setting_values.get("Pinhole"))
    refractive_index = str(setting_values.get("RefractionIndex", ""))
    for index, element in enumerate(channels):
        values = {str(key): str(value) for key, value in element.attrib.items()}
        lut_name = values.get("LUTName", "")
        values.update(
            {
                "Color": _lut_color(lut_name),
                "ColorMode": "BaseColor",
                "ColorOpacity": "1.000",
                "Description": "(description not specified)",
                "GammaCorrection": "1.000",
                "LSMEmissionWavelength": emissions.get(index, ""),
                "LSMExcitationWavelength": "",
                "LSMPinhole": pinhole,
                "Name": lut_name or f"Channel {index + 1}",
                "RefractionIndexImmersion": refractive_index,
            }
        )
        sections[f"Channel {index}"] = values

    if recording_date:
        sections["TimeInfo"] = {
            "DatasetTimePoints": "1",
            "FileTimePoints": "1",
            "TimePoint1": recording_date,
        }
    return sections


def _lif_extents(
    dimensions: Mapping[str, Mapping[str, str]],
    setting: Mapping[str, str],
) -> dict[str, str]:
    swap_xy = str(setting.get("SwapXY", "0")).strip() == "1"
    stage_names = {
        "X": "StagePosY" if swap_xy else "StagePosX",
        "Y": "StagePosX" if swap_xy else "StagePosY",
    }
    result: dict[str, str] = {}
    for axis, extent_index in {"X": 0, "Y": 1, "Z": 2}.items():
        values = dimensions.get(axis)
        if values is None:
            continue
        try:
            count = int(values["NumberOfElements"])
            origin = np.float32(float(values["Origin"]) * 1_000_000.0)
            length = np.float32(float(values["Length"]) * 1_000_000.0)
            if axis == "Z":
                stage = np.float32(0.0)
            else:
                stage = np.float32(float(setting[stage_names[axis]]) * 1_000_000.0)
            if count <= 1:
                step = np.float32(abs(length))
            else:
                step = np.float32(abs(length) / np.float32(count - 1))
            start = np.float32(stage + origin)
            end = np.float32(start + length)
            minimum = np.float32(min(start, end) - step)
            maximum = np.float32(max(start, end))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(np.isfinite(value) for value in (minimum, maximum)):
            continue
        result[f"ExtMin{extent_index}"] = format(float(minimum), ".9f")
        result[f"ExtMax{extent_index}"] = format(float(maximum), ".9f")
    return result


def _lif_recording_date(image: Any) -> str:
    try:
        timestamps = np.asarray(getattr(image, "timestamps", ())).reshape(-1)
    except Exception:
        return ""
    if not timestamps.size:
        return ""
    try:
        value = np.datetime64(timestamps[0]).astype("datetime64[ms]")
    except (TypeError, ValueError):
        return ""
    if np.isnat(value):
        return ""
    return np.datetime_as_string(value, unit="ms").replace("T", " ")


def _lif_emission_wavelengths(
    setting: ElementTree.Element | None,
) -> dict[int, str]:
    if setting is None:
        return {}
    spectro = _first_xml_element(setting, "Spectro")
    if spectro is None:
        return {}
    result = {}
    for element in _xml_elements(spectro, "MultiBand"):
        try:
            index = int(element.attrib["Channel"]) - 1
            left = float(element.attrib["LeftWorld"])
            right = float(element.attrib["RightWorld"])
        except (KeyError, TypeError, ValueError):
            continue
        midpoint = (left + right) / 2.0
        if index >= 0 and np.isfinite(midpoint):
            result[index] = format(midpoint, ".12f")
    return result


def _scaled_pinhole(value: Any) -> str:
    try:
        result = float(value) * 500_000.0
    except (TypeError, ValueError):
        return ""
    return format(result, ".12f") if np.isfinite(result) else ""


def _lut_color(name: str) -> str:
    colors = {
        "red": "1.000 0.000 0.000",
        "green": "0.000 1.000 0.000",
        "blue": "0.000 0.000 1.000",
        "cyan": "0.000 1.000 1.000",
        "magenta": "1.000 0.000 1.000",
        "yellow": "1.000 1.000 0.000",
        "gray": "1.000 1.000 1.000",
        "grey": "1.000 1.000 1.000",
    }
    return colors.get(str(name).strip().casefold(), "1.000 1.000 1.000")


def _require_complete_lif_metadata(
    source: Path,
    image: Any,
    sections: Mapping[str, Mapping[str, str]],
) -> None:
    missing = []
    image_values = sections.get("Image", {})
    if not sections.get("TimeInfo", {}).get("TimePoint1"):
        missing.append("the first acquisition timestamp")
    dims = tuple(str(value).upper() for value in getattr(image, "dims", ()))
    for axis, extent_index in (("X", 0), ("Y", 1), ("Z", 2)):
        if axis not in dims:
            continue
        if f"Dimension {axis}" not in sections:
            missing.append(f"DimensionDescription for {axis}")
        if not image_values.get(f"ExtMin{extent_index}") or not image_values.get(
            f"ExtMax{extent_index}"
        ):
            missing.append(f"the physical extent for {axis}")
    if missing:
        detail = ", ".join(missing)
        raise ImarisMetadataError(
            f"Cannot write a lossless IMS checkpoint from {source}: could not "
            f"decode {detail}."
        )


def _xml_elements(
    root: ElementTree.Element,
    local_name: str,
) -> list[ElementTree.Element]:
    return [
        element for element in root.iter() if _xml_local_name(element.tag) == local_name
    ]


def _first_xml_element(
    root: ElementTree.Element,
    local_name: str,
    *,
    required_attribute: str = "",
) -> ElementTree.Element | None:
    return next(
        (
            element
            for element in root.iter()
            if _xml_local_name(element.tag) == local_name
            and (not required_attribute or required_attribute in element.attrib)
        ),
        None,
    )


def _xml_local_name(value: Any) -> str:
    text = str(value)
    return text.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _container_values(container: Any) -> list[Any]:
    if container is None:
        return []
    if hasattr(container, "values"):
        return list(container.values())
    return list(container)


def _load_h5py():
    try:
        return import_module("h5py")
    except ImportError as exc:
        raise ImarisMetadataError(
            "Lossless IMS metadata preservation requires h5py. "
            'Install it with: pip install "napari-vipp[ims]"'
        ) from exc


def _load_liffile():
    try:
        return import_module("liffile")
    except ImportError as exc:
        raise ImarisMetadataError(
            "LIF metadata preservation requires liffile. "
            'Install it with: pip install "napari-vipp[microscope]"'
        ) from exc


def _decode_name(value: Any) -> str:
    return str(value).replace("%s", "/").replace("%p", "%")


def _decode_value(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, str):
        return value.rstrip("\x00")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "a"}:
            return value.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
        if value.dtype.kind == "U":
            return "".join(value.reshape(-1).tolist()).rstrip("\x00")
        if value.dtype.kind == "O":
            return "".join(_decode_value(item) for item in value.reshape(-1))
        if value.ndim == 0:
            return _decode_value(value.item())
        return json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, np.generic):
        return _decode_value(value.item())
    if value is None:
        return ""
    return str(value)


__all__ = [
    "ImarisMetadataError",
    "read_imaris_dataset_info",
    "read_lif_dataset_info",
]
