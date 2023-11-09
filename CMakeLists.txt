cmake_minimum_required(VERSION 3.12)
project(Denoiser)

find_package(CUDA REQUIRED)
include_directories(src/core)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=all -Wno-deprecated-gpu-targets)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ==========================================
# OpenEXR & zlib
find_package(ZLIB)
if(NOT ZLIB_FOUND)
  # Build zlib
  message(STATUS "Building zlib.")
  set(ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
  set(ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
  add_subdirectory(src/ext/zlib)

  set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/zlib")
  set(ZLIB_LIBRARY zlibstatic)
  set_property(TARGET zlibstatic PROPERTY FOLDER "ext")
  include_directories(${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/src/ext/zlib")
else()
  message(STATUS "zlib has been detected.")
endif()
include_directories(${ZLIB_INCLUDE_DIRS})


set(ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
set(OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
set(ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
set(OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)

add_subdirectory(src/ext/openexr)
set_property(TARGET IexMath eLut toFloat b44ExpLogTable dwaLookups CopyIlmBaseLibs IlmThread Half Iex Imath IlmImf IlmImf-obj PROPERTY FOLDER "ext")

include_directories (
    ${CMAKE_BINARY_DIR}/src/ext/openexr/IlmBase/config
    ${CMAKE_BINARY_DIR}/src/ext/openexr/OpenEXR/config
    src/ext/openexr/IlmBase/Half
    src/ext/openexr/IlmBase/Iex
    src/ext/openexr/IlmBase/Imath
    src/ext/openexr/OpenEXR/IlmImf
)
if(WIN32)
  set(OPENEXR_LIBRARIES IlmImf Imath Half ${ZLIB_LIBRARY})
else()
  set(OPENEXR_LIBRARIES IlmImf Imath Half)
endif()

source_group(ext REGULAR_EXPRESSION src/ext/.*)


# ==========================================
# DenoiserCore project
cuda_add_library(DenoiserCore STATIC
    src/core/denoiser.cu
    src/core/denoiser.h
)

# ==========================================
# DenoiserTester project
add_executable(DenoiserTester
    src/tester/tester.cpp
    src/tester/imageio.cpp
    src/tester/imageio.h
)

set_target_properties(DenoiserTester PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(DenoiserTester PRIVATE 
    DenoiserCore
    ${OPENEXR_LIBRARIES}
    ${ZLIB_LIBRARY}
)