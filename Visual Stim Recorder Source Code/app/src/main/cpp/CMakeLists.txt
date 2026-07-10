cmake_minimum_required(VERSION 3.22.1)
project("visualstimrecorder")

add_library(${PROJECT_NAME} SHARED
        native-lib.cpp)

# Include LSL headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Link prebuilt LSL
add_library(lsl SHARED IMPORTED)
set_target_properties(lsl PROPERTIES
        IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/liblsl.so
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        android
        log
        lsl)