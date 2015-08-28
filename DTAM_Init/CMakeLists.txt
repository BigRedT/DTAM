cmake_minimum_required ( VERSION 2.8 )
project ( correctRadialDistortion )

# Check if compiler supports ISO C++ 2011 standard and set it if it does
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# Third party libraries
find_package ( OpenCV REQUIRED )
find_package ( Ceres REQUIRED )

include_directories ( ${CERES_INCLUDE_DIRS} )

# Header files
set ( INCLUDE_DIR ${PROJECT_SOURCE_DIR} )
include_directories ( ${INCLUDE_DIR} )

add_executable ( LKTrackBA main.cpp LKTrackBA.cpp )
target_link_libraries ( LKTrackBA ${OpenCV_LIBS} ${CERES_LIBRARIES} ) 
