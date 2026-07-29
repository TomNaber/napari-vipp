from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from types import SimpleNamespace
from xml.etree import ElementTree

import dask.array as da
import numpy as np
import pytest

import napari_vipp.core.io.imaris as imaris_io
import napari_vipp.core.io.imaris_metadata as imaris_metadata_io
import napari_vipp.core.io.microscope as microscope_io
from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    build_batch_plan,
)
from napari_vipp.core.io import WRITE_FORMATS, write_image
from napari_vipp.core.io.imaris_metadata import (
    ImarisMetadataError,
    read_imaris_dataset_info,
    read_lif_dataset_info,
)
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
        self.application_name = application_name
        self.application_version = application_version
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
        source=SourceMetadata(uri="source.npy", format="numpy"),
        history=("Gaussian Blur",),
    )


def _set_ims_attribute(group, name: str, value: str) -> None:
    if name in group.attrs:
        del group.attrs[name]
    group.attrs[name] = np.frombuffer(value.encode("utf-8"), dtype="S1")


def _fake_lif_module():
    image_xml = ElementTree.fromstring(
        """
        <Element Name="3-1-4-13-G4_24-0034_A_8kHz">
          <Image TextDescription="" />
          <ChannelDescription DataType="0" ChannelTag="0" Resolution="8"
            NameOfMeasuredQuantity="" Min="0.000000e+000" Max="2.550000e+002"
            Unit="" LUTName="Red" IsLUTInverted="0" BytesInc="0" BitInc="0" />
          <ChannelDescription DataType="0" ChannelTag="0" Resolution="8"
            NameOfMeasuredQuantity="" Min="0.000000e+000" Max="2.550000e+002"
            Unit="" LUTName="Green" IsLUTInverted="0" BytesInc="1048576"
            BitInc="0" />
          <ChannelDescription DataType="0" ChannelTag="0" Resolution="8"
            NameOfMeasuredQuantity="" Min="0.000000e+000" Max="2.550000e+002"
            Unit="" LUTName="Blue" IsLUTInverted="0" BytesInc="2097152"
            BitInc="0" />
          <DimensionDescription DimID="1" NumberOfElements="1024"
            Origin="1.376765e-020" Length="8.201074e-005" Unit="m"
            BitInc="0" BytesInc="1" />
          <DimensionDescription DimID="2" NumberOfElements="1024"
            Origin="1.376765e-020" Length="8.201074e-005" Unit="m"
            BitInc="0" BytesInc="1024" />
          <DimensionDescription DimID="3" NumberOfElements="174"
            Origin="4.324700e-003" Length="-5.193460e-005" Unit="m"
            BitInc="0" BytesInc="3145728" />
          <Attachment Name="HardwareSetting" SystemTypeName="TCS SP8" />
          <ATLConfocalSettingDefinition StagePosX="0.0838162298652"
            StagePosY="0.0441872461392" SwapXY="1" Magnification="63"
            ObjectiveName="HC PL APO CS2    63x/1.40 OIL "
            MicroscopeModel="DMI6000B-CS" Immersion="OIL"
            NumericalAperture="1.4" RefractionIndex="1.518"
            Pinhole="9.55444839857651e-05">
            <Spectro>
              <MultiBand Channel="1" LeftWorld="562" RightWorld="760" />
              <MultiBand Channel="2" LeftWorld="760" RightWorld="790" />
            </Spectro>
          </ATLConfocalSettingDefinition>
          <LMSDataContainerHeader Version="2" />
        </Element>
        """
    )
    file_xml = ElementTree.fromstring(
        """
        <LMSDataContainerHeader Version="2">
          <Experiment />
        </LMSDataContainerHeader>
        """
    )
    experiment_path = (
        r"C:\Users\UMIC\UserData\2026\07_July\Tom Naber"
        r"\3-1-4 Batch 4.lif"
    )
    file_xml.find("Experiment").set("Path", experiment_path)
    data = da.zeros((174, 3, 1024, 1024), dtype=np.uint8, chunks=(1, 1, 16, 16))

    class FakeImage:
        name = "3-1-4-13-G4_24-0034_A_8kHz"
        shape = data.shape
        dtype = data.dtype
        dims = ("Z", "C", "Y", "X")
        timestamps = np.asarray(["2026-07-17T15:39:36.796"], dtype="datetime64[ms]")
        xml_element = image_xml

        def asxarray(self):
            return SimpleNamespace(data=data, dims=self.dims, attrs={}, coords={})

    class FakeLifFile:
        def __init__(self, _path):
            self.images = [FakeImage()]
            self.xml_element = file_xml

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    return SimpleNamespace(LifFile=FakeLifFile, __version__="2026.7.14")


def _install_fake_lif(monkeypatch):
    module = _fake_lif_module()
    monkeypatch.setattr(imaris_metadata_io, "_load_liffile", lambda: module)
    monkeypatch.setattr(
        microscope_io,
        "_optional_import",
        lambda name, _suffix: module if name == "liffile" else None,
    )
    return module


def test_ims_writer_streams_tczyx_blocks_and_metadata(monkeypatch, tmp_path):
    writer = _fake_writer()
    monkeypatch.setattr(imaris_io, "_load_writer", lambda: writer)
    expected = np.arange(2 * 2 * 17 * 3 * 5, dtype=np.uint16).reshape(2, 2, 17, 3, 5)
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


def test_ims_parameter_merge_preserves_vendor_metadata_and_updates_edits():
    data = np.zeros((1, 2, 3, 4, 5), dtype=np.uint16)
    state = replace(
        _explicit_state(data),
        channels=(
            ChannelMetadata("FITC", 0x00FF00, emission_wavelength=520.0),
            ChannelMetadata("DAPI", 0x0000FF, emission_wavelength=461.0),
        ),
        acquisition=AcquisitionMetadata(
            acquisition_date="2026-01-02T03:04:05Z",
            objective_na=1.4,
            refractive_index=1.515,
        ),
    )
    source = {
        "Image": {
            "OriginalFormat": "Leica: Image File Format LIF",
            "ExperimentPath": r"C:\Acquisition\original.lif",
            "RecordingDate": "2025-05-06 07:08:09.123",
            "NumericalAperture": "0.8",
            "NumberOfChannels": "3",
        },
        "Channel 0": {
            "Name": "DAPI",
            "LSMPinhole": "42.5",
            "RefractionIndexImmersion": "1.33",
            "DataType": "0",
            "Color": "0 0 1",
            "ColorMode": "BaseColor",
        },
        "Channel 1": {
            "Name": "FITC",
            "LSMPinhole": "55.5",
            "RefractionIndexImmersion": "1.33",
            "DataType": "0",
            "Color": "0 1 0",
            "ColorMode": "BaseColor",
        },
        "Channel 2": {"Name": "Discarded", "LSMPinhole": "99"},
        "Dimension X": {"NumberOfElements": "100"},
        "TimeInfo": {"TimePoint1": "2025-05-06 07:08:09.123"},
        "Log": {"Entries": "1", "Entry0": "original import"},
    }

    mapped = imaris_io._mapped_channel_parameters(state, 2, source)
    merged = imaris_io._merged_parameter_sections(
        state,
        {"t": 1, "c": 2, "z": 3, "y": 4, "x": 5},
        source,
        mapped,
    )

    assert merged["Image"]["OriginalFormat"] == source["Image"]["OriginalFormat"]
    assert merged["Image"]["ExperimentPath"] == source["Image"]["ExperimentPath"]
    assert merged["Image"]["NumericalAperture"] == "1.4"
    assert merged["Image"]["NumberOfChannels"] == "2"
    assert merged["Log"] == source["Log"]
    assert merged["Channel 0"]["LSMPinhole"] == "55.5"
    assert merged["Channel 0"]["LSMEmissionWavelength"] == "520"
    assert merged["Channel 1"]["LSMPinhole"] == "42.5"
    assert merged["Channel 1"]["RefractionIndexImmersion"] == "1.515"
    assert merged["Channel 1"]["DataType"] == "0"
    assert "Color" not in merged["Channel 1"]
    assert "Channel 2" not in merged
    assert merged["Dimension X"] == source["Dimension X"]
    assert "TimeInfo" not in merged


def test_lif_metadata_matches_imaris_convert_reference(
    monkeypatch,
    tmp_path,
):
    _install_fake_lif(monkeypatch)
    path = tmp_path / "3-1-4 Batch 4.lif"
    path.write_bytes(b"synthetic LIF fixture")

    sections = read_lif_dataset_info(path, require_complete=True)

    image = sections["Image"]
    assert image["RecordingDate"] == "2026-07-17 15:39:36.796"
    assert image["OriginalFormat"] == "Leica: Image File Format LIF"
    assert image["ElementName"] == "3-1-4-13-G4_24-0034_A_8kHz"
    assert image["ExperimentPath"] == (
        r"C:\Users\UMIC\UserData\2026\07_July\Tom Naber\3-1-4 Batch 4.lif"
    )
    assert image["MicroscopeModality"] == "TCS SP8"
    assert image["LensPower"] == "63"
    assert image["NumericalAperture"] == "1.4"
    assert image["RefractionIndex"] == "1.518"
    assert image["ExtMin0"] == "44187.164062500"
    assert image["ExtMax0"] == "44269.257812500"
    assert image["ExtMin1"] == "83816.148437500"
    assert image["ExtMax1"] == "83898.234375000"
    assert image["ExtMin2"] == "4272.465332031"
    assert image["ExtMax2"] == "4324.700195312"
    assert sections["TimeInfo"]["TimePoint1"] == image["RecordingDate"]
    assert sections["Channel 0"]["LSMPinhole"] == "47.772241992883"
    assert sections["Channel 0"]["LSMEmissionWavelength"] == "661.000000000000"
    assert sections["Channel 1"]["LSMEmissionWavelength"] == "775.000000000000"
    assert sections["Channel 1"]["BytesInc"] == "1048576"
    assert sections["Dimension Z"] == {
        "DimID": "3",
        "NumberOfElements": "174",
        "Origin": "4.324700e-003",
        "Length": "-5.193460e-005",
        "Unit": "m",
        "BitInc": "0",
        "BytesInc": "3145728",
    }


def test_lif_reader_uses_synthesized_imaris_metadata(monkeypatch, tmp_path):
    _install_fake_lif(monkeypatch)
    path = tmp_path / "3-1-4 Batch 4.lif"
    path.write_bytes(b"synthetic LIF fixture")

    loaded = microscope_io.read_microscope(path)
    state = loaded.image_state

    assert state.axis_order == "ZCYX"
    axes = {axis.name: axis for axis in state.axes}
    assert axes["x"].unit == "micrometer"
    assert axes["x"].translation == pytest.approx(44187.1640625)
    assert axes["y"].translation == pytest.approx(83816.1484375)
    assert axes["z"].translation == pytest.approx(4272.465332031)
    assert [channel.name for channel in state.channels] == ["Red", "Green", "Blue"]
    assert [channel.color for channel in state.channels] == [
        0xFF0000,
        0x00FF00,
        0x0000FF,
    ]
    assert state.channels[0].emission_wavelength == 661.0
    assert state.acquisition.acquisition_date == "2026-07-17 15:39:36.796"
    assert state.acquisition.instrument == "TCS SP8"
    assert state.acquisition.objective_na == 1.4
    assert state.acquisition.objective_magnification == 63.0
    assert state.acquisition.refractive_index == 1.518


def test_lif_ims_export_preserves_layout_and_uses_vipp_identity(
    monkeypatch,
    tmp_path,
):
    _install_fake_lif(monkeypatch)
    source = tmp_path / "3-1-4 Batch 4.lif"
    source.write_bytes(b"synthetic LIF fixture")
    data = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", "µm", 0.6, 4272.465332031),
            AxisMetadata("y", "space", "µm", 0.2, 83816.1484375),
            AxisMetadata("x", "space", "µm", 0.2, 44187.1640625),
        ),
        channels=(ChannelMetadata("A"), ChannelMetadata("B")),
        acquisition=AcquisitionMetadata(
            objective_na=1.49,
            refractive_index=1.48,
        ),
        source=SourceMetadata(
            uri=str(source),
            format="leica-lif",
            series_index=0,
        ),
    )
    writer = _fake_writer()
    monkeypatch.setattr(imaris_io, "_load_writer", lambda: writer)

    write_image(data, tmp_path / "checkpoint.ims", format="ims", image_state=state)

    converter = _ImageConverter.instances[-1]
    parameters = converter.finished[1].values
    assert converter.application_name == "napari-vipp"
    assert converter.application_version
    assert parameters[("Image", "RecordingDate")] == "2026-07-17 15:39:36.796"
    assert parameters[("Image", "OriginalFormat")] == ("Leica: Image File Format LIF")
    assert parameters[("Image", "ElementName")] == ("3-1-4-13-G4_24-0034_A_8kHz")
    assert parameters[("Channel 0", "DataType")] == "0"
    assert parameters[("Channel 0", "BytesInc")] == "0"
    assert parameters[("Channel 0", "Name")] == "A"
    assert parameters[("Channel 0", "RefractionIndexImmersion")] == "1.48"
    assert parameters[("Dimension X", "NumberOfElements")] == "1024"
    assert parameters[("Dimension X", "BytesInc")] == "1"
    assert parameters[("Image", "NumericalAperture")] == "1.49"
    assert len(converter.finished[2]) == 1


def test_lif_ims_export_rejects_missing_timestamp_before_writing(
    monkeypatch,
    tmp_path,
):
    module = _install_fake_lif(monkeypatch)
    original = module.LifFile

    class MissingTimestampLifFile(original):
        def __init__(self, path):
            super().__init__(path)
            self.images[0].timestamps = np.asarray([], dtype="datetime64[ms]")

    module.LifFile = MissingTimestampLifFile
    source = tmp_path / "missing-time.lif"
    source.write_bytes(b"synthetic LIF fixture")
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm"),
            AxisMetadata("y", "space", "µm"),
            AxisMetadata("x", "space", "µm"),
        ),
        source=SourceMetadata(uri=str(source), format="leica-lif"),
    )
    output = tmp_path / "must-not-exist.ims"

    with pytest.raises(ImarisMetadataError, match="first acquisition timestamp"):
        write_image(data, output, format="ims", image_state=state)

    assert not output.exists()


def test_imaris_dataset_info_reads_attributes_and_long_parameters(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "metadata.ims"
    with h5py.File(path, "w") as handle:
        root = handle.create_group("DataSetInfo")
        image = root.create_group("Image")
        image.attrs["OriginalFormat"] = np.frombuffer(b"Leica LIF", dtype="S1")
        custom = root.create_group("Custom%pSection")
        custom.create_dataset(
            "Long%sValue",
            data=np.frombuffer(b"complete parameter", dtype="S1"),
        )

    sections = read_imaris_dataset_info(path)

    assert sections["Image"]["OriginalFormat"] == "Leica LIF"
    assert sections["Custom%Section"]["Long/Value"] == "complete parameter"


def test_ims_export_fails_before_writing_if_source_metadata_is_unavailable(
    tmp_path,
):
    data = np.zeros((3, 4, 5), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm", 1.0),
            AxisMetadata("y", "space", "µm", 0.3),
            AxisMetadata("x", "space", "µm", 0.2),
        ),
        source=SourceMetadata(
            uri=str(tmp_path / "missing.ims"),
            format="imaris-ims+bioio",
        ),
    )
    output = tmp_path / "must-not-exist.ims"

    with pytest.raises(ImarisMetadataError, match="source is missing"):
        write_image(data, output, format="ims", image_state=state)

    assert not output.exists()


def test_in_place_ims_export_captures_metadata_before_replacing_source(
    monkeypatch,
    tmp_path,
):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "in-place.ims"
    with h5py.File(path, "w") as handle:
        image = handle.create_group("DataSetInfo").create_group("Image")
        _set_ims_attribute(image, "OriginalFormat", "Leica LIF")

    data = np.zeros((1, 2, 3, 4, 5), dtype=np.uint16)
    state = replace(
        _explicit_state(data),
        source=SourceMetadata(uri=str(path), format="imaris-ims+bioio"),
    )
    writer = _fake_writer()
    monkeypatch.setattr(imaris_io, "_load_writer", lambda: writer)

    write_image(data, path, format="ims", image_state=state)

    converter = _ImageConverter.instances[-1]
    parameters = converter.finished[1]
    assert parameters.values[("Image", "OriginalFormat")] == "Leica LIF"
    assert converter.application_name == ""
    assert converter.application_version == ""


def test_ims_export_rejects_malformed_source_metadata_before_writing(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "malformed.ims"
    with h5py.File(source, "w") as handle:
        handle.create_group("DataSet")
    data = np.zeros((3, 4, 5), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm", 1.0),
            AxisMetadata("y", "space", "µm", 0.3),
            AxisMetadata("x", "space", "µm", 0.2),
        ),
        source=SourceMetadata(uri=str(source), format="imaris-ims+bioio"),
    )
    output = tmp_path / "must-not-exist.ims"

    with pytest.raises(ImarisMetadataError, match="no /DataSetInfo"):
        write_image(data, output, format="ims", image_state=state)

    assert not output.exists()


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
def test_native_lif_ims_ims_checkpoint_does_not_need_original(
    monkeypatch,
    tmp_path,
):
    _install_fake_lif(monkeypatch)
    source = tmp_path / "3-1-4 Batch 4.lif"
    source.write_bytes(b"synthetic LIF fixture")
    data = np.arange(3 * 2 * 3 * 4, dtype=np.uint8).reshape(3, 2, 3, 4)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", "µm", 0.3002, 4272.465332031),
            AxisMetadata("y", "space", "µm", 0.08017, 83816.1484375),
            AxisMetadata("x", "space", "µm", 0.08017, 44187.1640625),
        ),
        channels=(
            ChannelMetadata("Red", 0xFF0000, emission_wavelength=661.0),
            ChannelMetadata("Green", 0x00FF00, emission_wavelength=775.0),
            ChannelMetadata("Blue", 0x0000FF),
        ),
        acquisition=AcquisitionMetadata(
            acquisition_date="2026-07-17 15:39:36.796",
            objective_na=1.4,
            refractive_index=1.518,
        ),
        source=SourceMetadata(uri=str(source), format="leica-lif"),
    )
    first = tmp_path / "lif-checkpoint.ims"
    second = tmp_path / "ims-checkpoint.ims"

    write_image(data, first, format="ims", image_state=state)
    from napari_vipp.core.io import read_image

    loaded = read_image(first)
    source.unlink()
    write_image(loaded.data, second, format="ims", image_state=loaded.image_state)

    reloaded = read_image(second)
    reloaded_data = (
        reloaded.data.compute() if hasattr(reloaded.data, "compute") else reloaded.data
    )
    np.testing.assert_array_equal(np.squeeze(np.asarray(reloaded_data)), data)
    sections = read_imaris_dataset_info(second)
    assert sections["Image"]["RecordingDate"] == "2026-07-17 15:39:36.796"
    assert sections["Image"]["OriginalFormat"] == ("Leica: Image File Format LIF")
    assert sections["Image"]["ElementName"] == "3-1-4-13-G4_24-0034_A_8kHz"
    assert sections["Image"]["ExperimentPath"].endswith("3-1-4 Batch 4.lif")
    assert sections["Image"]["OriginalFormatFileIOVersion"] == ("liffile 2026.7.14")
    assert sections["Channel 0"]["DataType"] == "0"
    assert sections["Channel 1"]["BytesInc"] == "1048576"
    assert sections["Dimension Z"]["NumberOfElements"] == "174"
    assert sections["ImarisDataSet"]["Creator"] == "napari-vipp"


@pytest.mark.skipif(
    importlib.util.find_spec("napari_vipp_imaris") is None,
    reason="native ImarisWriter wheel is not installed",
)
def test_native_ims_round_trip_through_bioformats(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm", 1.0),
            AxisMetadata("y", "space", "µm", 0.3),
            AxisMetadata("x", "space", "µm", 0.2),
        ),
    )
    path = tmp_path / "native-source.ims"

    write_image(data, path, format="ims", image_state=state)
    from napari_vipp.core.io import read_image

    with h5py.File(path, "r+") as handle:
        info = handle["DataSetInfo"]
        image = info["Image"]
        for name, value in {
            "OriginalFormat": "Leica: Image File Format LIF",
            "OriginalFormatFileIOVersion": "9.7.2",
            "ExperimentPath": r"C:\Acquisition\original.lif",
            "RecordingDate": "2025-05-06 07:08:09.123",
            "ExtMin0": "10",
            "ExtMax0": "11",
            "ExtMin1": "20",
            "ExtMax1": "21.2",
            "ExtMin2": "30",
            "ExtMax2": "33",
            "NumericalAperture": "0.8",
        }.items():
            _set_ims_attribute(image, name, value)
        channel = info["Channel 0"]
        for name, value in {
            "LSMEmissionWavelength": "461",
            "LSMPinhole": "47.772241992883",
            "RefractionIndexImmersion": "1.33",
        }.items():
            _set_ims_attribute(channel, name, value)
        _set_ims_attribute(info["Log"], "Entries", "1")
        _set_ims_attribute(info["Log"], "Entry0", "original acquisition log")
        _set_ims_attribute(
            info["TimeInfo"],
            "TimePoint1",
            "2025-05-06 07:08:09.123",
        )

    loaded = read_image(path)
    loaded_data = (
        loaded.data.compute() if hasattr(loaded.data, "compute") else loaded.data
    )

    np.testing.assert_array_equal(np.squeeze(np.asarray(loaded_data)), data)
    assert loaded.image_state.source.format == "imaris-ims+bioio"
    assert [axis.translation for axis in loaded.image_state.axes[-3:]] == [
        30.0,
        20.0,
        10.0,
    ]
    assert loaded.image_state.channels[0].emission_wavelength == 461.0
    assert loaded.image_state.acquisition.objective_na == 0.8
    assert loaded.image_state.acquisition.refractive_index == 1.33

    processed_state = replace(
        loaded.image_state,
        axes=tuple(
            replace(axis, scale={"x": 0.4, "y": 0.5, "z": 2.0}[axis.name])
            if axis.name in {"x", "y", "z"}
            else axis
            for axis in loaded.image_state.axes
        ),
        channels=(replace(loaded.image_state.channels[0], emission_wavelength=500.0),),
        acquisition=replace(
            loaded.image_state.acquisition,
            objective_na=1.49,
            refractive_index=1.518,
        ),
        history=loaded.image_state.history
        + ("Set Pixel Size", "Set Microscope Metadata"),
    )
    checkpoint = tmp_path / "checkpoint.ims"
    final = tmp_path / "final.ims"
    write_image(
        loaded.data,
        checkpoint,
        format="ims",
        image_state=processed_state,
    )
    checkpoint_loaded = read_image(checkpoint)
    path.unlink()
    write_image(
        checkpoint_loaded.data,
        final,
        format="ims",
        image_state=checkpoint_loaded.image_state,
    )

    final_loaded = read_image(final)
    final_data = (
        final_loaded.data.compute()
        if hasattr(final_loaded.data, "compute")
        else final_loaded.data
    )
    np.testing.assert_array_equal(np.squeeze(np.asarray(final_data)), data)
    assert np.asarray(final_data).dtype == data.dtype

    final_info = read_imaris_dataset_info(final)
    assert final_info["Image"]["OriginalFormat"] == ("Leica: Image File Format LIF")
    assert final_info["Image"]["OriginalFormatFileIOVersion"] == "9.7.2"
    assert final_info["Image"]["ExperimentPath"] == r"C:\Acquisition\original.lif"
    assert final_info["Image"]["RecordingDate"] == "2025-05-06 07:08:09.123"
    assert float(final_info["Image"]["ExtMin0"]) == pytest.approx(10.0)
    assert float(final_info["Image"]["ExtMax0"]) == pytest.approx(12.0)
    assert float(final_info["Image"]["ExtMin1"]) == pytest.approx(20.0)
    assert float(final_info["Image"]["ExtMax1"]) == pytest.approx(22.0)
    assert float(final_info["Image"]["ExtMin2"]) == pytest.approx(30.0)
    assert float(final_info["Image"]["ExtMax2"]) == pytest.approx(36.0)
    assert float(final_info["Image"]["NumericalAperture"]) == pytest.approx(1.49)
    assert final_info["Channel 0"]["LSMPinhole"] == "47.772241992883"
    assert float(final_info["Channel 0"]["LSMEmissionWavelength"]) == 500.0
    assert float(final_info["Channel 0"]["RefractionIndexImmersion"]) == (
        pytest.approx(1.518)
    )
    assert final_info["Log"]["Entry0"] == "original acquisition log"
