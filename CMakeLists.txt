cmake_minimum_required(VERSION 3.20)

project(snake VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(Pagmo CONFIG REQUIRED)

add_executable(snake)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp)
target_sources(snake PRIVATE ${sources})

target_include_directories(snake PRIVATE
    "${PROJECT_BINARY_DIR}/src"
    "${PROJECT_SOURCE_DIR}/src"
)

target_link_libraries(snake PRIVATE
    Threads::Threads
    Pagmo::pagmo
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(snake PRIVATE
        -Wall
        -Wstrict-null-sentinel
        -Wno-comment
        -Wno-unused-parameter
    )
endif()