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

target_link_libraries(${PROJECT_NAME}_TEST PRIVATE gtest ${PROJECT_NAME}_LIB)
target_include_directories(${PROJECT_NAME}_TEST PRIVATE gtest ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set_target_properties(${PROJECT_NAME}_TEST PROPERTIES DEBUG_POSTFIX "_d")