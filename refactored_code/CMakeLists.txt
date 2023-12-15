cmake_minimum_required(VERSION 3.10)

project(BlindDeblur)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    include/deconv_opts.h
    include/conv2.h
    include/fast_deconv_bregman.h
    include/ms_blind_deconv.h
    include/pcg_kernel_irls_conv.h
    include/solve_image_bregman.h
    include/ss_blind_deconv.h
    src/deconv_opts.cpp
    src/conv2.cpp
    src/fast_deconv_bregman.cpp
    src/ms_blind_deconv.cpp
    src/pcg_kernel_irls_conv.cpp
    src/solve_image_bregman.cpp
    src/ss_blind_deconv.cpp
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        include
        ${OpenCV_INCLUDE_DIRS}
)
