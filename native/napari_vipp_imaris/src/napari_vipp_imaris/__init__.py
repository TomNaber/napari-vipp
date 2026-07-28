"""Package-local loader for the bundled ImarisWriter library."""

from pathlib import Path

from . import PyImarisWriter as PyImarisWriter

__version__ = "0.1.0a1"


class _BundledImageConverter(PyImarisWriter.ImageConverter):
    def _get_dll_filename(self):
        return str(Path(__file__).with_name("libbpImarisWriter96.dylib"))


def _exception_init(self, text):
    Exception.__init__(self, text)


PyImarisWriter.ImageConverter = _BundledImageConverter
PyImarisWriter.PyImarisWriterException.__init__ = _exception_init

__all__ = ["PyImarisWriter", "__version__"]
