cmake_minimum_required(VERSION 3.10)
PROJECT(UDF)
if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
    message(STATUS "ascend compiler enter.")
    # if unsupport current resource type, please uncomment the next line.
    # message(FATAL_ERROR "unsupport compile Ascend target!")
elseif ("x${RESOURCE_TYPE}" STREQUAL "xAarch")
    message(STATUS "Aarch64 compiler enter.")
    # if unsupport current resource type, please uncomment the next line.
    # message(FATAL_ERROR "Unsupport compile aarch64 target!")
else ()
    message(STATUS "x86 compiler enter.")
    # if unsupport current resource type, please uncomment the next line.
    # message(FATAL_ERROR "Unsupport compile x86 target!")
endif ()

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_INSTALL_PATH $ENV{ASCEND_INSTALL_PATH})
    message(STATUS "Read ASCEND_INSTALL_PATH from ENV: ${ASCEND_INSTALL_PATH}")
else()
    set(ASCEND_INSTALL_PATH /user/local/Ascend)
    message(STATUS "Deafault ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}. You can export ASCEND_INSTALL_PATH")
endif()

# set dynamic library output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
# set static library output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})

message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

set(INC_DIR ${ASCEND_INSTALL_PATH}/latest/include)
file(GLOB SRC_LIST "*.cpp")

# Specify cross compiler
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Set C++ compiler
set(CMAKE_CXX_COMPILER ${TOOLCHAIN})

#========================UDF so compile========================================
# check if SRC_LIST is exist
if("x${SRC_LIST}" STREQUAL "x")
    message(UDF "=================no resource file")
    add_custom_target(${UDF_TARGET_LIB}
        COMMAND echo "no resource file to make lib${UDF_TARGET_LIB}.so")
endif()

# message(UDF "=================SRC_LIST=${SRC_LIST}=========================")
add_library(${UDF_TARGET_LIB} SHARED ${SRC_LIST})

target_include_directories(${UDF_TARGET_LIB} PRIVATE ${INC_DIR})

target_compile_options(${UDF_TARGET_LIB} PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)

if("x${RESOURCE_TYPE}" STREQUAL "xAscend")
    target_link_libraries(${UDF_TARGET_LIB} PRIVATE
        -Wl,--whole-archive
        ${ASCEND_INSTALL_PATH}/latest/aarch64-linux/lib64/stub/linux/aarch64/libflow_func.so
        -Wl,--no-whole-archive
    )
elseif("x${RESOURCE_TYPE}" STREQUAL "xAarch")
    target_link_libraries(${UDF_TARGET_LIB} PRIVATE
        -Wl,--whole-archive
        ${ASCEND_INSTALL_PATH}/latest/aarch64-linux/lib64/stub/linux/aarch64/libflow_func.so
        -Wl,--no-whole-archive
    )
else()
    target_link_libraries(${UDF_TARGET_LIB} PRIVATE
        -Wl,--whole-archive
        ${ASCEND_INSTALL_PATH}/latest/aarch64-linux/lib64/stub/linux/x86_64/libflow_func.so
        -Wl,--no-whole-archive
    )
endif()