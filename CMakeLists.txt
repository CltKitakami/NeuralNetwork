cmake_minimum_required(VERSION 2.8.8)

project(NeuralNetwork)


option(VERBOSE "VERBOSE" OFF)
option(BUILD_DEMO_APPS "BUILD_DEMO_APPS" ON)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release;Debug")
set_property(CACHE EXT_INCS PROPERTY STRINGS "")
set_property(CACHE EXT_LIBS PROPERTY STRINGS "")


if(VERBOSE)
	set(CMAKE_VERBOSE_MAKEFILE ON)
else()
	set(CMAKE_VERBOSE_MAKEFILE OFF)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	message("Build type: Debug")
else()
	message("Build type: Release")
	set(CMAKE_BUILD_TYPE "Release")
	add_definitions(-DNDEBUG)
endif()


set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib")


if(CMAKE_COMPILER_IS_GNUCXX)

	set(WARNINGS "-Wall -Wextra -Wunreachable-code -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wswitch-default -Wunused-parameter -Wreturn-type -Wsign-conversion -Wsign-promo -Werror=non-virtual-dtor")
	set(NOWARNINGS "-Wno-parentheses -Wno-unused-function -Wno-unused-label -Wno-unused-result")

	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

	set(CMAKE_CXX_FLAGS "-pipe ${WARNINGS} ${NOWARNINGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fopenmp")

	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
		set(CMAKE_EXE_LINKER_FLAGS "-s")
	endif()

elseif(MSVC)

	set(WARNINGS "/W4 /WX- /wd4127 /wd4481 /wd4512")
	set(CMAKE_CXX_FLAGS "${WARNINGS} /EHsc")

endif()


# Include path
include_directories("${CMAKE_SOURCE_DIR}")
include_directories(SYSTEM ${EXT_INCS})


# Subdirectories
add_subdirectory(core)


# Link libraries
add_library(${PROJECT_NAME} STATIC
	$<TARGET_OBJECTS:core>
)
target_link_libraries(${PROJECT_NAME} ${EXT_LIBS} ${SYSTEM_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)


# Build test executables
if(BUILD_DEMO_APPS)
	add_subdirectory(demo)
endif()
