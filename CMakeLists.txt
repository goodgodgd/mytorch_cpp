cmake_minimum_required(VERSION 3.15)
project(MLP
    VERSION 0.1.0
    DESCRIPTION "project to practice cmake"
    LANGUAGES CXX
)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_BUILD_TYPE Debug)
find_package(Eigen3 3.3 NO_MODULE)

# add_library(mytorch SHARED)
# target_sources(mytorch PRIVATE lib/mytorch/layers.cpp)
# target_include_directories(mytorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_compile_options(mytorch PRIVATE -Wall -Werror -O2)
# target_link_libraries(mytorch PRIVATE Eigen3::Eigen)
# set_property(TARGET mytorch PROPERTY POSITION_INDEPENDENT_CODE ON)
add_executable(mlp main.cpp)
target_compile_options(mlp PRIVATE -Wall -O2)

# target_link_libraries(mlp PRIVATE mytorch)
set_property(TARGET mlp PROPERTY CXX_STANDARD 17)
set_property(TARGET mlp PROPERTY CXX_STANDARD_REQUIRED On)
set_property(TARGET mlp PROPERTY CXX_EXTENSIONS Off)
