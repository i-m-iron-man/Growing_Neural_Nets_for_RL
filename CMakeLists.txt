cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${APP_NAME})


include_directories(${Boost_INCLUDE_DIRS})

set(USE_GL 1)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(MJC_path "/home/sid/library/mujoco/mujoco210/")
set(source_path "/home/sid/source/Growing_Nets/Source/meta_hebbian_cpp/")
include_directories(${MJC_path}/include ${source_path})

link_directories(${CMAKE_SOURCE_DIR}/home/sid/library/mujoco/mujoco210/bin/)
find_library(GLFW libglfw.so.3 HINTS /home/sid/library/mujoco/mujoco210/bin/)
message(STATUS "GLFW lib found at: " ${GLFW})

message(STATUS "MuJoCo path: " ${CMAKE_SOURCE_DIR}/home/sid/library/mujoco/mujoco210/bin/)

#Finding main mujoco library
if(${USE_GL})
file(GLOB LIB_MUJOCO /home/sid/library/mujoco/mujoco210/bin/libmujoco[0-9][0-9][0-9].so)
else()
file(GLOB LIB_MUJOCO /home/sid/library/mujoco/mujoco210/bin/libmujoco[0-9][0-9][0-9]nogl.so)
endif()
#Showing mujoco library found
message(STATUS "MuJoCo lib found at: " ${LIB_MUJOCO})

add_executable(${PROJECT_NAME} main.cpp)
target_sources(${PROJECT_NAME}
        PRIVATE
        env.h
        env.cpp
        help.h
        help.cpp
        Neuron.h
        Neuron.cpp
        Bond.h
        Bond.cpp
        Net.h
        Net.cpp
        Growing_Machines.h
        Growing_Machines.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${LIB_MUJOCO}
        ${GLFW}
        libGL.so
        libglew.so)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

