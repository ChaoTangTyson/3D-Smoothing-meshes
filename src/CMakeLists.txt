cmake_minimum_required(VERSION 3.1)
project(CourseWork2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"          ON)

find_package(LIBIGL REQUIRED QUIET)

# Add your project files
#file(GLOB SRC_FILES *.cpp)
include_directories(${PROJECT_SOURCE_DIR})

set(SRC_FILES ${PROJECT_SOURCE_DIR}/main.cpp
              ${PROJECT_SOURCE_DIR}/curvature.cpp
              ${PROJECT_SOURCE_DIR}/curvature.h
              ${PROJECT_SOURCE_DIR}/smoothing.cpp
              ${PROJECT_SOURCE_DIR}/smoothing.h
              )
              
              
add_executable(${PROJECT_NAME}_bin ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw  igl::opengl_glfw_imgui)
