#  FindSLIC.cmake
#  -- try to find SLIC package
#
#  SET: SLIC_SRC_DIR 
#
#  Once done, this will define
#    SLIC_FOUND - System has SLIC
#    SLIC_FILES - Files needed to use SLIC
#
#  Copyright (c) 2013 Grace Vesom
#

## SET SLIC SOURCE AND BUILD DIRS ##
SET(SLIC_SRC_DIR /home/grace/Projects/SLIC-Superpixels-OpenCV2.4/)

FILE(GLOB_RECURSE slic_source ${SLIC_SRC_DIR}/slic*.cpp)
FILE(GLOB_RECURSE slic_header ${SLIC_SRC_DIR}/slic*.h)

SET(SLIC_FILES ${slic_source} ${slic_header})

SET(SLIC_FOUND 0)

IF(SLIC_FILES)
  SET(SLIC_FOUND 1)
  MESSAGE(STATUS "Found SLIC -- " ${SLIC_SRC_DIR})
ELSE(SLIC_FILES)
  MESSAGE(FATAL_ERROR "Could not find SLIC")
ENDIF(SLIC_FILES)