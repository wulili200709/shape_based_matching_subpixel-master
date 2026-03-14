cmake_minimum_required(VERSION 3.16)
project(sbm_subpixel_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

set(SBM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../shape_based_matching-subpixel")
set(SBM_CUDA_ICP_ROOT "${SBM_ROOT}/cuda_icp")

pybind11_add_module(sbm_subpixel MODULE
    subpixel_pybind.cpp
    "${SBM_ROOT}/line2Dup.cpp"
    "${SBM_CUDA_ICP_ROOT}/icp.cpp"
    "${SBM_CUDA_ICP_ROOT}/scene/common.cpp"
    "${SBM_CUDA_ICP_ROOT}/scene/edge_scene/edge_scene.cpp"
)

target_include_directories(sbm_subpixel PRIVATE
    "${SBM_ROOT}"
    "${SBM_CUDA_ICP_ROOT}"
    "${SBM_CUDA_ICP_ROOT}/scene"
    "${SBM_CUDA_ICP_ROOT}/scene/edge_scene"
    "${OpenCV_INCLUDE_DIRS}"
)

target_compile_definitions(sbm_subpixel PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
)

# Apply /utf-8 only to the UTF-8 encoded file (subpixel_pybind.cpp);
# line2Dup.cpp is GBK encoded so must NOT get /utf-8.
if(MSVC)
    set_source_files_properties(subpixel_pybind.cpp PROPERTIES COMPILE_OPTIONS "/utf-8")
endif()

target_link_libraries(sbm_subpixel PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(sbm_subpixel PRIVATE OpenMP::OpenMP_CXX)
endif()

