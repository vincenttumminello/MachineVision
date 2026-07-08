# Injected by the MCHA4400 toolchain image.
#
# Ubuntu's libvtk9-dev ships a monolithic VTK-targets.cmake that references
# external imported targets for many VTK modules. Modern CMake validates those
# references at generate time, so find_package(VTK) can fail with
# "<Something>::<Target> ... but the target was not found" unless every
# referenced target exists. VTK resolves the dependencies of the *core* modules
# on its own, but the render/IO dependencies are only resolved when you request
# a rendering/IO component — so a project that asks for a different component
# set trips the check.
#
# This file is an official VTK config injection point (loaded by
# vtk-config.cmake before generation); project CMakeLists need no changes.
# It:
#   1. really finds the render/IO libraries, so they always exist as proper,
#      linkable targets no matter which VTK components you request; and
#   2. stubs the exotic externals (MPI, Qt, Python, HDF5, NetCDF, Ogg/Theora,
#      ...) that this course never links, with empty interface targets.
# Core dependencies that VTK always resolves itself (TBB, double-conversion,
# utf8cpp, Eigen3, zlib) are deliberately left untouched: pre-defining part of
# their multi-target export set would collide with VTK's own find_package.

# 1. Render/IO libraries VTK may link -- found for real (standard find modules,
#    idempotent, so a later find_package by VTK is harmless).
find_package(OpenGL QUIET)
find_package(GLEW QUIET)
find_package(Freetype QUIET)
find_package(X11 QUIET)
find_package(PNG QUIET)
find_package(JPEG QUIET)
find_package(TIFF QUIET)
find_package(EXPAT QUIET)
find_package(LibXml2 QUIET)
find_package(SQLite3 QUIET)
find_package(jsoncpp QUIET)

# 2. Externals for modules this course never links -- empty stubs are enough to
#    satisfy the generate-time check. Multi-target sets are stubbed in full.
foreach(_vtk_ext
        MPI::MPI_C
        Qt5::OpenGL Qt5::Widgets
        Python3::Module Python3::Python
        GL2PS::GL2PS
        NetCDF::NetCDF LibPROJ::LibPROJ LZ4::LZ4 LZMA::LZMA
        hdf5::hdf5 hdf5::hdf5_hl
        OGG::OGG THEORA::DEC THEORA::ENC THEORA::THEORA)
    if(NOT TARGET ${_vtk_ext})
        add_library(${_vtk_ext} INTERFACE IMPORTED)
    endif()
endforeach()
unset(_vtk_ext)
