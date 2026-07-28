"""Minimal installed-wheel smoke test."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from napari_vipp_imaris import PyImarisWriter as writer


def main() -> None:
    with TemporaryDirectory() as folder:
        path = Path(folder) / "smoke.ims"
        size = writer.ImageSize(x=4, y=3, z=2, c=1, t=1)
        one = writer.ImageSize(x=1, y=1, z=1, c=1, t=1)
        converter = writer.ImageConverter(
            "uint8",
            size,
            one,
            writer.DimensionSequence("x", "y", "z", "c", "t"),
            size,
            str(path),
            writer.Options(),
            "napari-vipp-imaris",
            "0.1.0a1",
            writer.CallbackClass(),
        )
        try:
            converter.CopyBlock(
                np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
                writer.ImageSize(x=0, y=0, z=0, c=0, t=0),
            )
            converter.Finish(
                writer.ImageExtents(0, 0, 0, 4, 3, 2),
                writer.Parameters(),
                [datetime(2026, 1, 1)],
                [writer.ColorInfo()],
                True,
            )
        finally:
            converter.Destroy()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("ImarisWriter did not create a usable IMS file.")


if __name__ == "__main__":
    main()
