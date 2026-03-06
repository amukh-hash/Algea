# ─────────────────────────────────────────────────────────────────────
# Algae Native Frontend — Root CMake Configuration
#
# Build targets:
#   - Algae_Live  (Release, dark palette, Algae_ENV_LIVE)
#   - Algae_Sim   (Release, watermark overlay, Algae_ENV_SIM)
#
# Dependencies managed via vcpkg manifest-mode (see vcpkg.json).
# Enforces: C++23, LTO, CRT linkage uniformity, TSAN for tests.
# ─────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.28)
project(AlgaeNativeFrontend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Enforce strict ISO C++23 (no GNU extensions)

# ── Link-Time Optimization (LTO / IPO) ────────────────────────────
# Blind Spot 3 Mitigation: LTO may strip Qt MOC symbols. QML_ELEMENT
# and QML_SINGLETON macros in headers force linker symbol preservation.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ── CRT Linkage Uniformity ────────────────────────────────────────
# Blind Spot 2 Mitigation: Force dynamic CRT (/MD) across all targets
# to match vcpkg's default. Prevents heap corruption when passing
# std::string / std::shared_ptr across DLL boundaries.
if(MSVC)
    set(VCPKG_CRT_LINKAGE dynamic)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# QT_NO_KEYWORDS prevents Qt from #define-ing `signals`, `slots`, `emit`
# as macros. This avoids collisions with Arrow's cancel.h which has a
# parameter named `signals`. Use Q_SIGNALS, Q_SLOTS, Q_EMIT instead.
add_compile_definitions(QT_NO_KEYWORDS)
# Silence C++20 deprecation of std::atomic_load_explicit for shared_ptr
add_compile_definitions(_SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING)

# ── Dependencies ───────────────────────────────────────────────────
find_package(Qt6 6.8 REQUIRED COMPONENTS Core Quick Qml Network Widgets Gui)
find_package(Vulkan REQUIRED)
find_package(Protobuf REQUIRED)
find_package(cppzmq REQUIRED)
find_package(Arrow REQUIRED)
find_package(Boost REQUIRED COMPONENTS headers)
find_package(GTest CONFIG)
find_package(OpenSSL QUIET)
find_package(mimalloc CONFIG QUIET)

# libfido2 — no CMake config, use find_library
find_library(FIDO2_LIBRARY NAMES fido2)
find_path(FIDO2_INCLUDE_DIR fido.h)

# moodycamel::ReaderWriterQueue (header-only)
find_path(READERWRITERQUEUE_INCLUDE_DIR readerwriterqueue/readerwriterqueue.h)

# KDDockWidgets
find_package(KDDockWidgets QUIET)

# ── Build Config Generation ──────────────────────────────────────
# Blind Spot 3 Mitigation: generate build_config.h via configure_file()
# bypassing MSVC cmd.exe quote-stripping on -D string defines.
configure_file(
    ${CMAKE_SOURCE_DIR}/src/config/build_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/build_config.h
    @ONLY
)

# libfido2 for FIDO2/CTAP2 hardware key auth
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBFIDO2 libfido2)
endif()

# ── Protobuf Code Generation ──────────────────────────────────────
file(GLOB PROTO_FILES "${CMAKE_SOURCE_DIR}/schemas/*.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# ── Source Files ───────────────────────────────────────────────────
set(ENGINE_SOURCES
    src/engine/ZmqReceiver.cpp
    src/engine/UiSynchronizer.cpp
    src/engine/AlertDag.cpp
    src/engine/GlobalStore.cpp
    src/engine/StateReconciler.cpp
)

set(MODEL_SOURCES
    src/models/ArrowTableModel.cpp
    src/models/ArrowBuilderWorker.cpp
)

set(RENDERING_SOURCES
    src/rendering/VulkanChartItem.cpp
    src/rendering/FanChartItem.cpp
    src/rendering/ParallelCoordinatesItem.cpp
    src/rendering/SankeyDiagramItem.cpp
)

set(NETWORK_SOURCES
    src/network/RestClient.cpp
)

set(HARDWARE_SOURCES
    src/hardware/KillSwitch.cpp
    src/hardware/KillSwitchBridge.cpp
    src/hardware/FidoGateway.cpp
)

set(WINDOWING_SOURCES
    src/windowing/WorkspaceManager.cpp
)

set(CONFIG_SOURCES
    src/config/BuildEnvironment.h
)

set(ALL_SOURCES
    src/main.cpp
    ${ENGINE_SOURCES}
    ${MODEL_SOURCES}
    ${RENDERING_SOURCES}
    ${NETWORK_SOURCES}
    ${HARDWARE_SOURCES}
    ${WINDOWING_SOURCES}
    ${CONFIG_SOURCES}
    ${PROTO_SRCS}
)

# ── QML Resources ─────────────────────────────────────────────────
qt_add_resources(QML_RESOURCES src/qml/qml.qrc)

# ── Target Configuration Macro ─────────────────────────────────────
# Unifies compiler flags and library linkage for both Live and Sim.
macro(configure_Algae_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${READERWRITERQUEUE_INCLUDE_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )

    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /O2 /arch:AVX2 /fp:fast /W4 /FS)
        # §2.1: mimalloc heap override — link-only mode.
        # The /FI approach conflicts with Arrow vendored headers.
        # Instead, mimalloc-redirect.dll intercepts at DLL import level.
        # Generate .pdb files in Release for MiniDump → WinDbg triage
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<CONFIG:Release>:/Zi>")
        # Control Flow Guard: prevents EDR false-positives on KillSwitch/SHM calls
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<CONFIG:Release>:/guard:cf>")
        target_link_options(${TARGET_NAME} PRIVATE
            "$<$<CONFIG:Release>:/DEBUG:FULL>"
            "$<$<CONFIG:Release>:/OPT:REF>"
            "$<$<CONFIG:Release>:/OPT:ICF>"
            "$<$<CONFIG:Release>:/guard:cf>"
        )
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -O3 -march=native -ffast-math -Wall -Wextra -Wpedantic)
    endif()

    target_link_libraries(${TARGET_NAME} PRIVATE
        Qt6::Core Qt6::Quick Qt6::Qml Qt6::Network Qt6::Widgets Qt6::Gui Qt6::Concurrent
        Vulkan::Vulkan
        protobuf::libprotobuf
        cppzmq
        Arrow::arrow_shared
        Boost::headers
    )

    # §2.1: mimalloc — thread-local lock-free allocator
    if(mimalloc_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE mimalloc)
    endif()

    if(FIDO2_LIBRARY AND FIDO2_INCLUDE_DIR)
        target_include_directories(${TARGET_NAME} PRIVATE ${FIDO2_INCLUDE_DIR})
        target_link_libraries(${TARGET_NAME} PRIVATE ${FIDO2_LIBRARY})
        target_compile_definitions(${TARGET_NAME} PRIVATE HAS_LIBFIDO2=1)
        if(MSVC)
            # Windows USB HID (SetupAPI) + CNG cryptography (bcrypt, crypt32)
            target_link_libraries(${TARGET_NAME} PRIVATE setupapi bcrypt crypt32)
        endif()
        if(OpenSSL_FOUND)
            target_link_libraries(${TARGET_NAME} PRIVATE OpenSSL::Crypto OpenSSL::SSL)
        endif()
    endif()

    if(KDDockWidgets_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE KDAB::kddockwidgets)
        target_compile_definitions(${TARGET_NAME} PRIVATE HAS_KDDOCKWIDGETS=1)
    endif()
endmacro()

# ── Live Target ────────────────────────────────────────────────────
add_executable(Algae_Live ${ALL_SOURCES} ${QML_RESOURCES})
target_compile_definitions(Algae_Live PRIVATE Algae_ENV_LIVE=1)
configure_Algae_target(Algae_Live)

# ── Sim Target ─────────────────────────────────────────────────────
add_executable(Algae_Sim ${ALL_SOURCES} ${QML_RESOURCES})
target_compile_definitions(Algae_Sim PRIVATE Algae_ENV_SIM=1)
configure_Algae_target(Algae_Sim)

# ── Tests ──────────────────────────────────────────────────────────
if(GTest_FOUND)
    enable_testing()
    
    add_executable(Algae_tests
        tests/test_ring_buffer.cpp
        tests/test_alert_dag.cpp
        tests/test_arrow_model.cpp
        tests/test_killswitch_latency.cpp
        src/engine/AlertDag.cpp
        src/models/ArrowTableModel.cpp
        src/hardware/KillSwitch.cpp
        ${PROTO_SRCS}
    )
    target_include_directories(Algae_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${READERWRITERQUEUE_INCLUDE_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(Algae_tests PRIVATE
        GTest::gtest GTest::gtest_main
        Qt6::Core Qt6::Network
        protobuf::libprotobuf
        Arrow::arrow_shared
        Boost::headers
    )
    
    # Thread Sanitizer for mathematically proving atomic correctness
    if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(Algae_tests PRIVATE -fsanitize=thread)
        target_link_options(Algae_tests PRIVATE -fsanitize=thread)
    endif()
    
    include(GoogleTest)
    gtest_discover_tests(Algae_tests)
endif()

# ── Automated Deployment: windeployqt POST_BUILD ──────────────────
# Harvests all required Qt DLLs, Vulkan scene graph plugins, and QML
# imports into the output directory. Without this, standalone deployment
# to terminals without Qt SDK will show a white screen (missing qtvulkan.dll).
find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${Qt6_DIR}/../../../bin")

if(WINDEPLOYQT_EXECUTABLE AND WIN32)
    macro(add_windeployqt TARGET_NAME)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                --no-translations
                --no-system-d3d-compiler
                --no-opengl-sw
                --qmldir "${CMAKE_SOURCE_DIR}/src/qml"
                "$<TARGET_FILE:${TARGET_NAME}>"
            COMMENT "Running windeployqt for ${TARGET_NAME}..."
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        )
    endmacro()

    add_windeployqt(Algae_Live)
    add_windeployqt(Algae_Sim)
endif()
