cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

project(SpineMeasureAssist)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org")
set(EXTENSION_CONTRIBUTORS "Masaki Sakamoto (Kyoto University)")
set(EXTENSION_DESCRIPTION "Manual landmark-based measurement of spinal alignment parameters on lateral spine X-ray.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SpineMeasureAssist.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SpineMeasureAssist/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(LumbarMeasureAssist)
add_subdirectory(WholeSpineAssist)
add_subdirectory(CervicalMeasureAssist)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
