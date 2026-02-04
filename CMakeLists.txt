cmake_minimum_required(VERSION 3.18)
project(funclcms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(connected_component cpp/connected_component.cpp)
target_compile_options(connected_component PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>
)

install(TARGETS connected_component LIBRARY DESTINATION .)
