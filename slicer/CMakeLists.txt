cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

project(SpineMeasureAssist)

set(EXTENSION_HOMEPAGE "https://github.com/masaki39/spine-measure-assist")
set(EXTENSION_CATEGORY "Spine")
set(EXTENSION_CONTRIBUTORS "masaki39")
set(EXTENSION_DESCRIPTION "Spine alignment measurement tools for 3D Slicer")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(LumbarMeasureAssist)
add_subdirectory(CervicalMeasureAssist)
add_subdirectory(WholeSpineAssist)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
