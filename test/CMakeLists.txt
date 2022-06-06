include(FetchContent)
FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG release-1.11.0
)

FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SRC "*.cpp")
file(GLOB_RECURSE TEST_HEAD "*.h")
list(FILTER TEST_HEAD EXCLUDE REGEX ".*pch.h$")

add_executable(${PROJECT_NAME}_TEST ${TEST_SRC} ${TEST_HEAD})

target_precompile_headers(${PROJECT_NAME}_TEST PRIVATE "pch.h")


set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native)

target_link_libraries(${PROJECT_NAME}_TEST PRIVATE gtest ${PROJECT_NAME}_LIB /usr/lib/libgmpxx.so)
target_include_directories(${PROJECT_NAME}_TEST PRIVATE gtest ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src /usr/include)

set_target_properties(${PROJECT_NAME}_TEST PROPERTIES DEBUG_POSTFIX "_d")