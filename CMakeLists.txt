cmake_minimum_required(VERSION 2.8)

ADD_DEFINITIONS(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS "project src dir -- " ${CMAKE_MODULE_PATH})

FIND_PACKAGE(SLIC REQUIRED)

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

project( slic )
add_executable( slic test_slic.cpp ${SLIC_FILES})
target_link_libraries( slic ${OpenCV_LIBRARIES} )