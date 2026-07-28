from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest

import napari_vipp.core.io.imaris as imaris_io
from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    build_batch_plan,
)
from napari_vipp.core.io import WRITE_FORMATS, write_image
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
    SourceMetadata,
    image_state_from_array,
)


@dataclass
class _Size:
    x: int = 0
    y: int = 0
    z: int = 0
    c: int = 0
    t: int = 0


class _Parameters:
    def __init__(self):
        self.values = {}

    def set_value(self, section, name, value):
        self.values[(section, name)] = value

    def set_channel_name(self, index, name):
        self.set_value(f"Channel {index}", "Name", name)


class _ColorInfo:
    def set_base_color(self, color):
        self.color = color


class _ImageConverter:
    instances = []

    def __init__(
        self,
        datatype,
        image_size,
        sample_size,
        dimension_sequence,
        block_size,
        output_filename,
        options,
        application_name,
        application_version,
        callback,
    ):
        self.datatype = datatype
        self.image_size = image_size
        self.block_size = block_size
        self.blocks = []
        self.finished = None
        self.destroyed = False
        self.instances.append(self)
        with open(output_filename, "wb") as stream:
            stream.write(b"fake ims")

    def NeedCopyBlock(self, block_index):
        return True

    def CopyBlock(self, data, block_index):
        self.blocks.append((data.copy(), block_index))

    def Finish(self, extents, parameters, times, colors, adjust):
        self.finished = (extents, parameters, times, colors, adjust)

    def Destroy(self):
        self.destroyed = True


def _fake_writer():
    _ImageConverter.instances.clear()
    return SimpleNamespace(
        ImageSize=_Size,
        DimensionSequence=lambda *values: values,
        Options=type("Options", (), {}),
        CallbackClass=type("CallbackClass", (), {}),
        ImageConverter=_ImageConverter,
        ImageExtents=lambda *values: values,
        Parameters=_Parameters,
        ColorInfo=_ColorInfo,
        Color=lambda *values: values,
    )


def _explicit_state(data):
    return image_state_from_array(
        data,
        axes=(
            AxisMetadata("t", "time", "s", 2.0),
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", "nm", 1_000.0),
            AxisMetadata("y", "space", "µm", 0.3, 2.0),
            AxisMetadata("x", "space", "µm", 0.2, 10.0),
        ),
        source_name="Processed IMS",
        channels=(
            ChannelMetadata("DAPI", 0x0000FF),
            ChannelMetadata("FITC", 0x00FF00),
        ),
        acquisition=AcquisitionMetadata(acquisition_date="2026-01-02T03:04:05Z"),
        source=SourceMetadata(uri="source.ims", format="imaris-ims"),
        history=("Gaussian Blur",),
    )


def test_ims_writer_streams_tczyx_blocks_and_metadata(monkeypatch, tmp_path):
    writer = _fake_writer()
    monkeypatch.setattr(imaris_io, "_load_writer", lambda: writer)
    expected = np.arange(2 * 2 * 17 * 3 * 5, dtype=np.uint16).reshape(
        2, 2, 17, 3, 5
    )
    data = da.from_array(expected, chunks=(1, 1, 8, 3, 5))

    saved = write_image(
        data,
        tmp_path / "processed",
        format="ims",
        image_state=_explicit_state(data),
    )

    assert "ims" in WRITE_FORMATS
    assert saved == tmp_path / "processed.ims"
    converter = _ImageConverter.instances[-1]
    assert converter.datatype == "uint16"
    assert converter.destroyed
    assert len(converter.blocks) == 8
    rebuilt = np.zeros_like(expected)
    for block, index in converter.blocks:
        z0 = index.z * 16
        z1 = min(z0 + 16, expected.shape[2])
        rebuilt[index.t, index.c, z0:z1] = block[0, 0, : z1 - z0]
    np.testing.assert_array_equal(rebuilt, expected)

    extents, parameters, times, colors, adjust = converter.finished
    assert extents == pytest.approx((10.0, 2.0, 0.0, 11.0, 2.9, 17.0))
    assert parameters.values[("VIPP", "AxisOrder")] == "TCZYX"
    assert parameters.values[("Channel 0", "Name")] == "DAPI"
    assert (times[1] - times[0]).total_seconds() == 2.0
    assert len(colors) == 2
    assert adjust is True


def test_ims_writer_rejects_implicit_conversion_and_inferred_axes(tmp_path):
    uint64_data = np.zeros((3, 4), dtype=np.uint64)
    explicit = image_state_from_array(
        uint64_data,
        axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space")),
    )
    with pytest.raises(ValueError, match="does not silently convert"):
        write_image(
            uint64_data,
            tmp_path / "bad.ims",
            format="ims",
            image_state=explicit,
        )

    inferred = image_state_from_array(np.zeros((3, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="explicit semantics"):
        write_image(
            np.zeros((3, 4), dtype=np.uint8),
            tmp_path / "ambiguous.ims",
            format="ims",
            image_state=inferred,
        )


def test_ims_writer_reports_optional_install_command(monkeypatch):
    def missing(_name):
        raise ImportError("missing")

    monkeypatch.setattr(imaris_io, "import_module", missing)
    with pytest.raises(
        imaris_io.OptionalImarisWriterError,
        match=r"napari-vipp\[ims\]",
    ):
        imaris_io._load_writer()


def test_batch_plan_adds_ims_suffix(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "field.npy", np.zeros((2, 3), dtype=np.uint8))
    config = BatchConfig(
        workflow_file=tmp_path / "workflow.json",
        workflow_sha256="0" * 64,
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=(
            BatchOutputConfig(
                "output",
                "Batch Output",
                "processed",
                "image",
                "ims",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="ims",
    )

    plan = build_batch_plan(config)

    assert plan.items[0].outputs[0].path.name == "field__processed.ims"


@pytest.mark.skipif(
    importlib.util.find_spec("napari_vipp_imaris") is None,
    reason="native ImarisWriter wheel is not installed",
)
def test_native_ims_round_trip_through_bioformats(tmp_path):
    data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm", 1.0),
            AxisMetadata("y", "space", "µm", 0.3),
            AxisMetadata("x", "space", "µm", 0.2),
        ),
    )
    path = tmp_path / "native.ims"

    write_image(data, path, format="ims", image_state=state)
    from napari_vipp.core.io import read_image

    loaded = read_image(path)
    loaded_data = (
        loaded.data.compute() if hasattr(loaded.data, "compute") else loaded.data
    )

    np.testing.assert_array_equal(np.squeeze(np.asarray(loaded_data)), data)
    assert loaded.image_state.source.format == "imaris-ims+bioio"
