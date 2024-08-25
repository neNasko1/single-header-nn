# Copied straight from https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB test_SRC "*.h" "*.cpp")

message(${test_SRC})
add_executable(test_all ${test_SRC} "../src/small-nn.h")

target_link_libraries(test_all GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_all)
