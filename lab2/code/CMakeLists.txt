cmake_minimum_required(VERSION 3.10)
project(DecisionTree)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")



if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
    MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY)
    set (MKL_FIND_QUIETLY TRUE)
endif()

if(NOT BUILD_SHARED_LIBS)
    set(INT_LIB "libmkl_intel_ilp64.a")
    set(SEQ_LIB "libmkl_sequential.a")
    set(THR_LIB "libmkl_intel_thread.a")
    set(COR_LIB "libmkl_core.a")
else()
    set(INT_LIB "mkl_intel_ilp64")
    set(SEQ_LIB "mkl_sequential")
    set(THR_LIB "mkl_intel_thread")
    set(COR_LIB "mkl_core")
endif()

set(MKLROOT "/opt/intel/compilers_and_libraries_2018.3.185/mac/mkl")
set(INTELROOT "/opt/intel")

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKLROOT}/include)

find_library(MKL_INTERFACE_LIBRARY
    NAMES ${INT_LIB}
    PATHS ${MKLROOT}/lib
    ${MKLROOT}/lib/intel64
    ${INTELROOT}/mkl/lib/intel64
    NO_DEFAULT_PATH)

find_library(MKL_SEQUENTIAL_LAYER_LIBRARY
    NAMES ${SEQ_LIB}
    PATHS ${MKLROOT}/lib
    ${MKLROOT}/lib/intel64
    ${INTELROOT}/mkl/lib/intel64
    NO_DEFAULT_PATH)

find_library(MKL_CORE_LIBRARY
    NAMES ${COR_LIB}
    PATHS ${MKLROOT}/lib
    ${MKLROOT}/lib/intel64
    ${INTELROOT}/mkl/lib/intel64
    NO_DEFAULT_PATH)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
set(MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_SEQUENTIAL_LAYER_LIBRARY} ${MKL_CORE_LIBRARY})

if (MKL_INCLUDE_DIR AND
    MKL_INTERFACE_LIBRARY AND
    MKL_SEQUENTIAL_LAYER_LIBRARY AND
    MKL_CORE_LIBRARY)

    if (NOT DEFINED ENV{CRAY_PRGENVPGI} AND
        NOT DEFINED ENV{CRAY_PRGENVGNU} AND
        NOT DEFINED ENV{CRAY_PRGENVCRAY} AND
        NOT DEFINED ENV{CRAY_PRGENVINTEL})
        set(ABI "-m64")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMKL_ILP64 ${ABI}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 ${ABI}")

else()

    set(MKL_INCLUDE_DIRS "")
    set(MKL_LIBRARIES "")
    set(MKL_INTERFACE_LIBRARY "")
    set(MKL_SEQUENTIAL_LAYER_LIBRARY "")
    set(MKL_CORE_LIBRARY "")

endif()

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS MKL_INTERFACE_LIBRARY MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY)

MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_INTERFACE_LIBRARY MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY)

include_directories(${MKL_INCLUDE_DIRS})
include_directories(inc)
# include_directories(/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Headers)
# include_directories(/usr/local/lib/python2.7/site-packages/numpy/core/include)

# find_package( PythonInterp 2.7 REQUIRED )
# find_package( PythonLibs 2.7 REQUIRED )
find_package(Range-v3)
# find_package( PythonLibs 3 REQUIRED )
# include_directories( ${PYTHON_INCLUDE_DIRS} )

# find_package( Boost COMPONENTS python37 REQUIRED )

message(STATUS ${RANGE_V3_INCLUDE_DIRS})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS ${MKL_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp -DEIGEN_DONT_ALIGN_STATICALLY")
message(STATUS ${MKLROOT}/lib/libmkl_intel_lp64.a)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL  -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -liomp5 -lpthread -lm -ldl -m64")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib -lpython2.7")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_executable(main main.cpp inc/decision_tree.h inc/AI_utility.h src/decision_tree.cpp inc/array_view.h inc/matrix_view.h src/AI_utility.cpp src/matrix_view.cpp src/random_lib.cpp)
target_link_libraries (main Eigen3::Eigen gtest range-v3)
add_subdirectory(unit_test)
file(GLOB DATAFILES
    "data/*")
foreach(DATAFILE ${DATAFILES})
    get_filename_component(DATAFILENAME ${DATAFILE} NAME)
    file(COPY ${DATAFILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
endforeach()


