cmake_minimum_required(VERSION 3.18)
project(aspher CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native for local builds; OFF for redistributable wheels (SKBUILD)
if(NOT DEFINED ASPHER_NATIVE)
  if(SKBUILD)
    set(ASPHER_NATIVE OFF)
  else()
    set(ASPHER_NATIVE ON)
  endif()
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(ASPHER_NATIVE)
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
endif()

# Let an activated conda env (fenicsx-env) provide Eigen and Python.
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

find_package(Eigen3 3.4 QUIET)
if(NOT TARGET Eigen3::Eigen)
  # header-only fallback for isolated builds (e.g. pip) without system Eigen
  include(FetchContent)
  FetchContent_Declare(eigen3_src
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
  FetchContent_GetProperties(eigen3_src)
  if(NOT eigen3_src_POPULATED)
    FetchContent_Populate(eigen3_src)
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen INTERFACE ${eigen3_src_SOURCE_DIR})
endif()
find_package(OpenMP)

# FFT engine for the spectral preconditioner. Default: the bundled pocketfft
# (third_party/, BSD-3-Clause, header-only) — keeps binaries permissively
# licensed. -DASPHER_USE_FFTW=ON switches to FFTW3 plans (slightly faster;
# FFTW is GPL, so distributed binaries then carry GPL terms).
option(ASPHER_USE_FFTW "Use FFTW3 (GPL) instead of bundled pocketfft (BSD)" OFF)
if(ASPHER_USE_FFTW)
  find_path(FFTW3_INCLUDE_DIR fftw3.h)
  find_library(FFTW3_LIB fftw3)
  find_library(FFTW3F_LIB fftw3f)
  if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIB OR NOT FFTW3F_LIB)
    message(FATAL_ERROR "ASPHER_USE_FFTW=ON but FFTW3 (double + float) was "
                        "not found: install it, e.g. 'apt install libfftw3-dev' "
                        "or 'conda install fftw', or point CMAKE_PREFIX_PATH "
                        "at an FFTW installation.")
  endif()
  find_library(FFTW3_OMP_LIB fftw3_omp)
  find_library(FFTW3F_OMP_LIB fftw3f_omp)
endif()

add_library(aspher_core STATIC
  src/boussinesq_kernel.cpp
  src/cluster_tree.cpp
  src/hmatrix.cpp
  src/contact_solver.cpp
  src/cheb_basis.cpp
  src/uniform_quadtree.cpp
  src/h2_operator.cpp
  src/fourier_precond.cpp
  src/fft_operator.cpp
  src/nested_solve.cpp
)
target_include_directories(aspher_core PUBLIC include)
target_link_libraries(aspher_core PUBLIC Eigen3::Eigen)
if(ASPHER_USE_FFTW)
  target_include_directories(aspher_core PUBLIC ${FFTW3_INCLUDE_DIR})
  target_compile_definitions(aspher_core PUBLIC HMC_USE_FFTW)
  if(FFTW3_OMP_LIB AND FFTW3F_OMP_LIB)
    target_link_libraries(aspher_core PUBLIC ${FFTW3_OMP_LIB} ${FFTW3F_OMP_LIB})
    target_compile_definitions(aspher_core PUBLIC HMC_FFTW_THREADS)
  endif()
  target_link_libraries(aspher_core PUBLIC ${FFTW3_LIB} ${FFTW3F_LIB})
else()
  target_include_directories(aspher_core SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/pocketfft)
endif()
target_compile_options(aspher_core PRIVATE -Wall -Wextra)
set_target_properties(aspher_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
  target_link_libraries(aspher_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(NOT SKBUILD)
  enable_testing()
  foreach(t kernel hmatrix contact h2 precond fft active)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE aspher_core)
    target_compile_options(test_${t} PRIVATE -Wall -Wextra)
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
  # masked-matvec gate benchmark (not a test; driven by bench_masked_gate.py)
  add_executable(bench_masked tests/bench_masked.cpp)
  target_link_libraries(bench_masked PRIVATE aspher_core)
  target_compile_options(bench_masked PRIVATE -Wall -Wextra)
endif()

if(SKBUILD)
  # wheel build (scikit-build-core): pybind11 comes from the build env
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(aspher python/bindings.cpp)
  target_link_libraries(aspher PRIVATE aspher_core)
  install(TARGETS aspher LIBRARY DESTINATION .)
  install(FILES python/hmatrix_contact.py DESTINATION .) # compat alias
else()
  # dev build: pybind11 is not installed in fenicsx-env; pass its cmake dir
  # explicitly, e.g.
  #   -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/python/bindings.cpp)
    pybind11_add_module(aspher python/bindings.cpp)
    target_link_libraries(aspher PRIVATE aspher_core)
    set_target_properties(aspher PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  endif()
endif()
