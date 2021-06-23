cmake_minimum_required(VERSION 3.0)

project(main CXX) 
set(TARGET main)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR "D:\\Program Files\\opencv\\build")
find_package(OpenCV 3.4 REQUIRED)
include_directories(${NCNN_DIR}/include)

if(WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi  /Od /Bt")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG")
  set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /DEBUG")
endif()


# 1. path to Paddle-Lite lib and mklml lib
set(INFER_LIB "D:\\zhangjun31\\Paddle-Lite\\build.lite.x86.vs2015\\inference_lite_lib")
set(LITE_DIR "${INFER_LIB}/cxx")
set(MKLML_DIR "${INFER_LIB}/third_party/mklml/")

if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  set(MSVC_STATIC_CRT OFF)
  if(MSVC_STATIC_CRT)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
  else(MSVC_STATIC_CRT)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MDd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MD")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MD")
  endif(MSVC_STATIC_CRT)
endif()

# 2. link mklml and Paddle-Lite directory
link_directories(${LITE_DIR}/lib ${MKLML_DIR}/lib)
include_directories(${LITE_DIR}/include/ ${MKLML_DIR}/include)

# 3. compile options 
if (NOT WIN32)
  add_definitions(-std=c++11 -g -O3 -pthread)
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
endif()

# 4.add executable output
add_executable(${TARGET} ${TARGET}.cc)
if (WIN32)
  set(WITH_STATIC_MKL ON)
  if(WITH_STATIC_MKL)
    set(MATH_LIB ${MKLML_DIR}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${MKLML_DIR}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
                 ${MKLML_DIR}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()

  target_link_libraries(${TARGET} libpaddle_api_light_bundled.lib)
  target_link_libraries(${TARGET} shlwapi.lib)
  target_link_libraries(${TARGET} ${MATH_LIB})
  target_link_libraries(${TARGET} ${OpenCV_LIBS})

  add_custom_command(TARGET ${TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_DIR}/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
  )
  if(NOT WITH_STATIC_MKL)
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_DIR}/lib/mklml.dll ${CMAKE_BINARY_DIR}/Release
    )
  endif()
else()
    target_link_libraries(${TARGET} -lpaddle_light_api_shared)
    target_link_libraries(${TARGET} -liomp5)
    target_link_libraries(${TARGET} -ldl)
endif()
