# Injected by the MCHA4400 toolchain image.
#
# Ubuntu's libvtk9-dev is a "kitchen-sink" build: it ships every VTK module,
# including ones that link exotic externals this course never uses (Qt5, MPI,
# HDF5, NetCDF, Python3, Theora/Ogg, GL2PS, LibPROJ, ...). Two consequences that
# this file works around, in order:
#
#   1. A bare `find_package(VTK REQUIRED)` -- no COMPONENTS, as the lab template
#      uses -- makes Ubuntu's VTK config resolve EVERY module's external
#      dependency as REQUIRED. With those exotic -dev packages absent, configure
#      dies on the first one (find_package(Qt5), then MPI, then HDF5, ...). We
#      fix this by pinning a default component list (below) when the project
#      didn't specify one, so VTK resolves only the modules the course actually
#      uses plus their transitive module deps -- all of whose externals ARE
#      installed. VTK_LIBRARIES then contains just that sane set.
#
#   2. VTK-targets.cmake references imported targets for render/IO deps that must
#      exist at generate time. We find those for real, and keep empty stubs for
#      the exotic externals as a defensive backstop (harmless: only created if
#      the real target is absent, and those modules are never linked anyway).
#
# This file is an official VTK config injection point: vtk-config.cmake includes
# it (OPTIONAL) after VTK_FIND_COMPONENTS is set but before the module
# find-packages run, so setting VTK_FIND_COMPONENTS here takes effect. Project
# CMakeLists need no changes.

# 1. Pin a default component set for the no-COMPONENTS case. A project that DOES
#    pass COMPONENTS is left untouched. These are the modules the course's
#    visualisation code uses; VTK expands them to their transitive module deps.
if (NOT VTK_FIND_COMPONENTS)
  set(VTK_FIND_COMPONENTS
      CommonCore CommonDataModel CommonColor CommonComputationalGeometry
      FiltersSources FiltersGeneral FiltersModeling FiltersExtraction FiltersGeometry
      RenderingCore RenderingOpenGL2 RenderingFreeType RenderingAnnotation
      RenderingContext2D RenderingContextOpenGL2 ChartsCore
      InteractionStyle InteractionWidgets
      IOImage IOGeometry IOPLY IOXML IOLegacy
      ImagingCore)
endif ()

# 2. Render/IO libraries VTK may link -- found for real (standard find modules,
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

# 3. Defensive stubs for exotic externals of modules this course never links --
#    empty interface targets satisfy any generate-time target reference. Multi-
#    target sets are stubbed in full. (With the component list above these modules
#    aren't requested, so this is belt-and-suspenders.) Core dependencies VTK
#    resolves itself (TBB, double-conversion, utf8cpp, Eigen3, zlib) are left
#    untouched: pre-defining part of their export set would collide with VTK.
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
