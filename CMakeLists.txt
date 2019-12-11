cmake_minimum_required(VERSION 3.10.2)

#! BASE GRIDDER PROJECT
project(gridder C)

#! FLAGS FOR COMPILER
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

add_executable(gridder gridder.c)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} )
target_link_libraries(gridder ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARY} m)