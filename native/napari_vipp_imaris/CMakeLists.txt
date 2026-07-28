cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum supported macOS")
project(napari_vipp_imaris LANGUAGES C CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  message(FATAL_ERROR "napari-vipp-imaris currently supports macOS on Apple Silicon only")
endif()

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "@loader_path")

add_subdirectory(vendor/ImarisWriter EXCLUDE_FROM_ALL)
add_custom_target(napari_vipp_imaris_native ALL DEPENDS bpImarisWriter96)

install(
  TARGETS bpImarisWriter96
  LIBRARY DESTINATION napari_vipp_imaris
)
install(
  FILES
    vendor/ImarisWriter/python/PyImarisWriter/ImarisWriterCtypes.py
    vendor/ImarisWriter/python/PyImarisWriter/PyImarisWriter.py
    vendor/ImarisWriter/LICENSE.txt
  DESTINATION napari_vipp_imaris
)
