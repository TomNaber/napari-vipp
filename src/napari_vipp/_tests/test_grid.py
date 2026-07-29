from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.grid import (
    ImageGrid,
    compare_aligned_grids,
    compare_mask_broadcast_grids,
    compare_psf_sampling,
    validate_aligned_image_states,
    validate_mask_broadcast_image_states,
    validate_psf_image_states,
    validate_spatial_mask_crop_image_states,
)
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload


def _state(shape, axes):
    return image_state_from_array(
        np.zeros(shape, dtype=np.float32),
        axes=tuple(axes),
    )


def test_aligned_grid_accepts_default_and_explicit_pixel_coordinates():
    inferred = _state(
        (8, 9),
        (
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    pixels = _state(
        (8, 9),
        (
            AxisMetadata("y", "space", unit="pixels"),
            AxisMetadata("x", "space", unit="px"),
        ),
    )

    result = compare_aligned_grids(
        ImageGrid.from_image_state(inferred),
        ImageGrid.from_image_state(pixels),
    )

    assert result.compatible


def test_aligned_grid_compares_convertible_physical_units():
    micrometers = _state(
        (8, 9),
        (
            AxisMetadata(
                "y",
                "space",
                unit="micrometer",
                scale=0.1,
                translation=0.2,
            ),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    nanometers = _state(
        (8, 9),
        (
            AxisMetadata("y", "space", unit="nm", scale=100, translation=200),
            AxisMetadata("x", "space", unit="nanometer", scale=100),
        ),
    )

    result = compare_aligned_grids(
        ImageGrid.from_image_state(micrometers),
        ImageGrid.from_image_state(nanometers),
    )

    assert result.compatible


@pytest.mark.parametrize(
    ("candidate_axes", "expected_code"),
    [
        (
            (
                AxisMetadata("y", "space", unit="micrometer", scale=0.2),
                AxisMetadata("x", "space", unit="micrometer", scale=0.1),
            ),
            "scale",
        ),
        (
            (
                AxisMetadata("y", "space", unit="micrometer", scale=0.1),
                AxisMetadata("x", "space", unit="micrometer", scale=0.1),
            ),
            "unit",
        ),
        (
            (
                AxisMetadata(
                    "y",
                    "space",
                    unit="micrometer",
                    scale=0.1,
                    translation=1,
                ),
                AxisMetadata("x", "space", unit="micrometer", scale=0.1),
            ),
            "translation",
        ),
        (
            (
                AxisMetadata("x", "space", unit="micrometer", scale=0.1),
                AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            ),
            "axis_semantics",
        ),
    ],
)
def test_aligned_grid_detects_calibration_and_semantic_mismatches(
    candidate_axes,
    expected_code,
):
    reference = _state(
        (8, 9),
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    if expected_code == "unit":
        candidate_axes = (
            AxisMetadata("y", "space", scale=0.1),
            AxisMetadata("x", "space", scale=0.1),
        )
    candidate = _state((8, 9), candidate_axes)

    result = compare_aligned_grids(
        ImageGrid.from_image_state(reference),
        ImageGrid.from_image_state(candidate),
    )

    assert not result.compatible
    assert expected_code in {issue.code for issue in result.issues}


def test_aligned_grid_validation_names_inputs_and_requires_explicit_resampling():
    reference = _state(
        (8, 9),
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    candidate = _state(
        (8, 9),
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Add cannot combine Signal and Background.*sample spacing differs.*"
            "explicitly resample"
        ),
    ):
        validate_aligned_image_states(
            (reference, candidate),
            input_labels=("Signal", "Background"),
            operation_title="Add",
        )


def _tzyx_image_state(shape=(2, 2, 3, 4)):
    return _state(
        shape,
        (
            AxisMetadata("t", "time", unit="second", scale=2),
            AxisMetadata("z", "space", unit="micrometer", scale=0.5),
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
    )


def _tyx_mask_state(shape=(2, 3, 4), *, time_scale=2):
    return _state(
        shape,
        (
            AxisMetadata("t", "time", unit="second", scale=time_scale),
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
    )


def test_mask_grid_maps_tyx_to_tzyx_by_explicit_semantics_not_equal_sizes():
    compatibility = compare_mask_broadcast_grids(
        ImageGrid.from_image_state(_tzyx_image_state()),
        ImageGrid.from_image_state(_tyx_mask_state()),
    )

    assert compatibility.compatible
    assert compatibility.mask_to_image_axes == (0, 2, 3)


def test_mask_grid_rejects_inferred_mismatched_rank_semantics():
    inferred_mask = image_state_from_array(np.zeros((2, 3, 4), dtype=bool))

    with pytest.raises(
        ValueError,
        match="requires explicit axis semantics.*inferred semantics",
    ):
        validate_mask_broadcast_image_states(
            _tzyx_image_state(),
            inferred_mask,
        )


def test_mask_grid_rejects_calibration_mismatch_on_semantic_axis():
    with pytest.raises(ValueError, match="sample spacing differs"):
        validate_mask_broadcast_image_states(
            _tzyx_image_state(),
            _tyx_mask_state(time_scale=3),
        )


def test_psf_grid_allows_different_extent_and_origin_at_matching_sampling():
    image = _state(
        (3, 64, 64),
        (
            AxisMetadata("t", "time", unit="second", scale=2),
            AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    psf = _state(
        (9, 11),
        (
            AxisMetadata("y", "space", unit="nm", scale=100, translation=-400),
            AxisMetadata("x", "space", unit="nm", scale=100, translation=-500),
        ),
    )

    result = compare_psf_sampling(
        ImageGrid.from_image_state(image),
        ImageGrid.from_image_state(psf),
        spatial_ndim=2,
    )

    assert result.compatible


def test_psf_grid_accepts_matching_default_uncalibrated_sampling():
    image = image_state_from_array(np.zeros((64, 64), dtype=np.float32))
    psf = image_state_from_array(np.zeros((9, 9), dtype=np.float32))

    result = compare_psf_sampling(
        ImageGrid.from_image_state(image),
        ImageGrid.from_image_state(psf),
        spatial_ndim=2,
    )

    assert result.compatible


def test_psf_grid_does_not_invent_spacing_when_psf_calibration_is_missing():
    image = _state(
        (64, 64),
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    uncalibrated_psf = image_state_from_array(
        np.zeros((9, 9), dtype=np.float32)
    )

    validate_psf_image_states(
        image,
        uncalibrated_psf,
        spatial_ndim=2,
        operation_title="Richardson-Lucy Deconvolution",
    )


def test_psf_grid_rejects_sampling_mismatch_without_implicit_resampling():
    image = _state(
        (64, 64),
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    psf = _state(
        (9, 9),
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.05),
            AxisMetadata("x", "space", unit="micrometer", scale=0.05),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Richardson-Lucy Deconvolution cannot combine Image and PSF.*"
            "sample spacing differs.*does not resample PSFs implicitly"
        ),
    ):
        validate_psf_image_states(
            image,
            psf,
            spatial_ndim=2,
            operation_title="Richardson-Lucy Deconvolution",
        )


def test_pipeline_rejects_equal_shape_images_on_different_physical_grids():
    first = np.ones((8, 9), dtype=np.float32)
    second = np.ones((8, 9), dtype=np.float32)
    first_state = _state(
        first.shape,
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    second_state = _state(
        second.shape,
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    add = pipeline.add_node("add_images")
    pipeline.connect("input", add.id, target_port=0)
    pipeline.connect(second_source.id, add.id, target_port=1)

    with pytest.raises(
        ValueError,
        match="Add cannot combine Input 1 and Input 2.*sample spacing differs",
    ):
        pipeline.run(
            first,
            source_payloads={
                "input": SourcePayload(first, image_state=first_state),
                second_source.id: SourcePayload(second, image_state=second_state),
            },
        )


def test_pipeline_preserves_equal_shape_uncalibrated_image_math():
    first = np.ones((8, 9), dtype=np.float32)
    second = np.full((8, 9), 2, dtype=np.float32)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    add = pipeline.add_node("add_images")
    pipeline.connect("input", add.id, target_port=0)
    pipeline.connect(second_source.id, add.id, target_port=1)

    outputs = pipeline.run(
        first,
        source_payloads={
            "input": SourcePayload(first),
            second_source.id: SourcePayload(second),
        },
    )

    np.testing.assert_array_equal(outputs[add.id], np.full((8, 9), 3))


def test_pipeline_broadcasts_tyx_mask_over_tzyx_by_axis_semantics():
    image = np.empty((2, 2, 3, 4), dtype=np.int16)
    for time_index in range(2):
        for z_index in range(2):
            image[time_index, z_index] = 10 * time_index + z_index + 1
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, :, 0] = True
    mask[1, :, -1] = True

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    mask_source = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    pipeline.set_param(masked.id, "outside_value", -5)
    pipeline.connect(mask_source.id, threshold.id)
    pipeline.connect("input", masked.id, target_port=0)
    pipeline.connect(threshold.id, masked.id, target_port=1)

    outputs = pipeline.run(
        image,
        source_payloads={
            "input": SourcePayload(image, image_state=_tzyx_image_state()),
            mask_source.id: SourcePayload(mask, image_state=_tyx_mask_state()),
        },
    )

    expected_mask = np.broadcast_to(mask[:, None, :, :], image.shape)
    expected = np.where(expected_mask, image, -5)
    np.testing.assert_array_equal(outputs[masked.id], expected)


def test_pipeline_rejects_inferred_mismatched_rank_mask():
    image = np.ones((2, 2, 3, 4), dtype=np.float32)
    mask = np.ones((2, 3, 4), dtype=bool)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    mask_source = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    pipeline.connect(mask_source.id, threshold.id)
    pipeline.connect("input", masked.id, target_port=0)
    pipeline.connect(threshold.id, masked.id, target_port=1)

    with pytest.raises(
        ValueError,
        match="Mask Image cannot combine Image and Mask.*explicit axis semantics",
    ):
        pipeline.run(
            image,
            source_payloads={
                "input": SourcePayload(image, image_state=_tzyx_image_state()),
                mask_source.id: SourcePayload(mask),
            },
        )


def test_pipeline_preserves_same_shape_mask_behavior_without_explicit_axes():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    mask = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
        ]
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    mask_source = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    pipeline.connect(mask_source.id, threshold.id)
    pipeline.connect("input", masked.id, target_port=0)
    pipeline.connect(threshold.id, masked.id, target_port=1)

    outputs = pipeline.run(
        image,
        source_payloads={
            "input": SourcePayload(image),
            mask_source.id: SourcePayload(mask),
        },
    )

    np.testing.assert_array_equal(outputs[masked.id], np.where(mask, image, 0))


def _tczyx_image_state(shape=(2, 3, 4, 6, 7)):
    return image_state_from_array(
        np.zeros(shape, dtype=np.uint16),
        axes=(
            AxisMetadata("t", "time", unit="second", scale=2, translation=3),
            AxisMetadata("c", "channel"),
            AxisMetadata(
                "z", "space", unit="micrometer", scale=0.5, translation=4
            ),
            AxisMetadata(
                "y", "space", unit="micrometer", scale=0.2, translation=10
            ),
            AxisMetadata(
                "x", "space", unit="micrometer", scale=0.3, translation=20
            ),
        ),
        source_name="source volume",
        channels=tuple(ChannelMetadata(name=f"channel {index}") for index in range(3)),
    )


def _yx_roi_state(shape=(6, 7), *, y_scale=0.2, x_translation=20):
    return image_state_from_array(
        np.zeros(shape, dtype=np.uint16),
        axes=(
            AxisMetadata(
                "y", "space", unit="micrometer", scale=y_scale, translation=10
            ),
            AxisMetadata(
                "x",
                "space",
                unit="micrometer",
                scale=0.3,
                translation=x_translation,
            ),
        ),
        source_name="drawn ROI labels",
    )


def test_spatial_crop_grid_maps_yx_and_zyx_onto_tczyx():
    image_state = _tczyx_image_state()
    yx_state = _yx_roi_state()
    zyx_state = _state(
        (4, 6, 7),
        (
            AxisMetadata(
                "z", "space", unit="micrometer", scale=0.5, translation=4
            ),
            AxisMetadata(
                "y", "space", unit="micrometer", scale=0.2, translation=10
            ),
            AxisMetadata(
                "x", "space", unit="micrometer", scale=0.3, translation=20
            ),
        ),
    )

    assert validate_spatial_mask_crop_image_states(image_state, yx_state) == (3, 4)
    assert validate_spatial_mask_crop_image_states(image_state, zyx_state) == (
        2,
        3,
        4,
    )


def test_pipeline_crops_yx_roi_and_shifts_only_spatial_origins():
    image = np.arange(2 * 3 * 4 * 6 * 7, dtype=np.uint16).reshape(2, 3, 4, 6, 7)
    mask = np.zeros((6, 7), dtype=np.uint16)
    mask[1:5, 2:4] = 9
    mask[3:5, 2:6] = 9
    original_mask = mask.copy()
    image_state = _tczyx_image_state()
    mask_state = _yx_roi_state()
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    mask_source = pipeline.add_node("input")
    masked = pipeline.add_node("mask_image")
    pipeline.set_param(masked.id, "outside_value", 99)
    pipeline.set_param(masked.id, "crop_image_to_mask", True)
    assert pipeline.connect("input", masked.id, target_port=0).success
    assert pipeline.connect(mask_source.id, masked.id, target_port=1).success

    output = pipeline.run(
        image,
        source_payloads={
            "input": SourcePayload(image, image_state=image_state),
            mask_source.id: SourcePayload(mask, image_state=mask_state),
        },
    )[masked.id]

    expected = image[:, :, :, 1:5, 2:6].copy()
    expected[..., mask[1:5, 2:6] == 0] = 99
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(mask, original_mask)
    assert output.shape == (2, 3, 4, 4, 4)
    assert output.flags.c_contiguous

    output_state = pipeline.output_states[masked.id]
    assert output_state is not None
    assert [axis.translation for axis in output_state.axes] == [
        3,
        0,
        4,
        pytest.approx(10.2),
        pytest.approx(20.6),
    ]
    assert [axis.scale for axis in output_state.axes] == [2, 1, 0.5, 0.2, 0.3]
    assert output_state.channels == image_state.channels
    assert output_state.acquisition == image_state.acquisition
    assert output_state.source == image_state.source
    assert output_state.history[-1] == (
        "Mask Image: applied mask from 'drawn ROI labels'; cropped "
        "Y[1:5], X[2:6]"
    )


@pytest.mark.parametrize(
    ("mask_state", "message"),
    [
        (_yx_roi_state(shape=(5, 7)), "sizes differ"),
        (_yx_roi_state(y_scale=0.4), "sample spacing differs"),
        (_yx_roi_state(x_translation=21), "origins differ"),
    ],
)
def test_spatial_crop_grid_rejects_grid_mismatch(mask_state, message):
    with pytest.raises(ValueError, match=message):
        validate_spatial_mask_crop_image_states(_tczyx_image_state(), mask_state)


def test_spatial_crop_grid_rejects_ambiguous_image_axes():
    image_state = _state(
        (6, 6, 7),
        (
            AxisMetadata("y", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )

    with pytest.raises(ValueError, match="multiple y:space axes"):
        validate_spatial_mask_crop_image_states(image_state, _yx_roi_state())


def test_pipeline_rejects_psf_sampling_mismatch_before_deconvolution():
    image = np.zeros((16, 16), dtype=np.float32)
    image[8, 8] = 1
    psf = np.ones((3, 3), dtype=np.float32)
    image_state = _state(
        image.shape,
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.1),
            AxisMetadata("x", "space", unit="micrometer", scale=0.1),
        ),
    )
    psf_state = _state(
        psf.shape,
        (
            AxisMetadata("y", "space", unit="micrometer", scale=0.05),
            AxisMetadata("x", "space", unit="micrometer", scale=0.05),
        ),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
    pipeline.connect("input", deconvolution.id, target_port=0)
    pipeline.connect(psf_source.id, deconvolution.id, target_port=1)

    with pytest.raises(
        ValueError,
        match=(
            "Richardson-Lucy Deconvolution cannot combine Image and PSF.*"
            "does not resample PSFs implicitly"
        ),
    ):
        pipeline.run(
            image,
            source_payloads={
                "input": SourcePayload(image, image_state=image_state),
                psf_source.id: SourcePayload(psf, image_state=psf_state),
            },
        )
