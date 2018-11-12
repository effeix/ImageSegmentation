cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(MAIN_SOURCES main.cpp)
set(MAIN_EXE_NAME <executable_name>)

find_package(Boost REQUIRED COMPONENTS <boost_components>)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${Boost_INCLUDE_DIR}
)

file(GLOB SOURCES *.cpp)

add_executable(
    ${MAIN_EXE_NAME}
    ${MAIN_SOURCES}
    ${SOURCES}
)

target_link_libraries(
    ${MAIN_EXE_NAME}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
)
