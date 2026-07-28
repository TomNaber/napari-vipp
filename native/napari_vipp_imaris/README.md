# napari-vipp-imaris

Optional macOS-arm64 native writer used by napari-vipp for direct `.ims`
export. The wheel bundles Oxford Instruments' Apache-2.0 ImarisWriter at
commit `b128e6e7d1a147261e9d5caf24ebc6b5c9c63779` plus its HDF5, zlib, and LZ4
runtime dependencies.

Build dependencies are CMake, HDF5, zlib, and LZ4. End users should install a
prebuilt wheel through `pip install "napari-vipp[ims]"`.

