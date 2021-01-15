cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project( normal_distributions_transform )

set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNING 1 CACHE INTERNAL "No dev warnings" )
endif()

FIND_PACKAGE(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(normal_distributions_transform normal_distributions_transform.cpp)
target_link_libraries(normal_distributions_transform ${PCL_LIBRARIES})

