cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project( Iterative_Closest_Points )

set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNING 1 CACHE INTERNAL "No dev warnings" )
endif()

FIND_PACKAGE(PCL 1.8 REQUIRED COMPONENT common io registration visualization features filters)

include_directories( ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# 使用ICP迭代最近点算法
add_executable (iterative_closest_point iterative_closest_point.cpp)
target_link_libraries (iterative_closest_point ${PCL_LIBRARIES})

# 使用迭代最近点算法，逐步实现地对一系列点云进行两两匹配
add_executable (pairwise_icp pairwise_incremental_registration.cpp)
target_link_libraries (pairwise_icp ${PCL_LIBRARIES})

# 交互式ICP可视化的程序
add_executable(iterative_icp_space iterative_icp_space.cpp)
target_link_libraries(iterative_icp_space ${PCL_LIBRARIES})

