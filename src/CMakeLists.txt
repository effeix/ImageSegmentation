cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED)
find_package(PkgConfig REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${Boost_INCLUDE_DIR}
)

# ---

# add_executable(
#     imageseg
#     main.cpp
# )

# target_link_libraries(
#     imageseg
#     ${Boost_LIBRARIES}
# )

# ---

add_executable(
    imageseg_seq
    main.cpp
    imagem.cpp
)
