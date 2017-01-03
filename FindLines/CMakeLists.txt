cmake_minimum_required(VERSION 3.0)
project(main)

set(CMAKE_CXX_FLAGS "-Wno-c++11-extensions -std=c++11 -Wno-deprecated-register")
find_package(OpenCV REQUIRED)
find_package( Boost COMPONENTS system filesystem REQUIRED )

include_directories(~/stash/third_party/)
include_directories(/usr/local/include)
include_directories(/usr/local/Cellar/eigen/3.2.8/include/eigen3)
include_directories(/opt/local/include)

add_executable(find_lines find_lines.cc)
target_link_libraries(find_lines 
                      ${OpenCV_LIBS}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY})
