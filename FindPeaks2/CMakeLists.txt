cmake_minimum_required(VERSION 3.3)
project(ConvertImage)

find_package( OpenCV REQUIRED)
find_package( Boost COMPONENTS system filesystem REQUIRED )

include_directories( 
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# add the executable
add_executable( Main Main.cpp ReadFile.cpp)

# add libraries
target_link_libraries( Main 
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${OpenCV_LIBS}
)