cmake_minimum_required(VERSION 3.22)
project(SchroedingerApproximation CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_DEBUG_POSTFIX _d)

# including source files
file(GLOB_RECURSE SRC "src/*.cpp")

file(GLOB_RECURSE HEAD "src/*.h")

list(FILTER HEAD EXCLUDE REGEX ".*pch.h$")

include(FetchContent)

# building executable
add_executable(${PROJECT_NAME} ${SRC} ${HEAD})

target_precompile_headers(${PROJECT_NAME} PUBLIC "src/sa/pch.h")

# building library for tests
add_library(${PROJECT_NAME}_LIB ${SRC} ${HEAD})

target_compile_definitions(${PROJECT_NAME}_LIB PRIVATE -D SA_LIB)
target_precompile_headers(${PROJECT_NAME}_LIB PUBLIC "src/sa/pch.h")

add_subdirectory("test")