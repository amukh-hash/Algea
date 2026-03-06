// ─────────────────────────────────────────────────────────────────────
// Logger — spdlog Async Lock-Free Logging
//
// Replaces synchronous qDebug()/std::cout with an 8192-message
// lock-free circular ring buffer. The ingestion thread pushes log
// strings in O(1) and immediately returns. The background thread
// flushes to disk independently.
//
// Overflow policy: overrun_oldest — drops old logs rather than
// blocking the critical path of the trading engine.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <memory>
#include <string>

#ifdef HAS_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#endif

namespace algae::engine {

inline void initializeAsyncLogger(const std::string& log_dir = "C:\\Algae\\Logs") {
#ifdef HAS_SPDLOG
    // 1. 8192-message lock-free queue with 1 background flush thread
    spdlog::init_thread_pool(8192, 1);

    // 2. Rotating file sink: 50MB per file, 5 rotated files retained
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_dir + "\\algae_native.log", 1024 * 1024 * 50, 5
    );

    // 3. Console sink for development builds
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::warn);

    // 4. Async logger with overrun_oldest — never blocks the calling thread
    auto async_logger = std::make_shared<spdlog::async_logger>(
        "algae",
        spdlog::sinks_init_list{file_sink, console_sink},
        spdlog::thread_pool(),
        spdlog::async_overflow_policy::overrun_oldest
    );

    // 5. Only force OS-level fsync on severe errors to prevent disk I/O saturation
    async_logger->flush_on(spdlog::level::err);

    spdlog::set_default_logger(async_logger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [T%t] %v");

    SPDLOG_INFO("Algae async logger initialized (8192-message ring buffer)");
#endif
}

inline void shutdownLogger() {
#ifdef HAS_SPDLOG
    spdlog::shutdown();
#endif
}

} // namespace algae::engine
