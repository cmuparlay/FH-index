#pragma once

// Logging macros
#ifdef VERBOSE
#define LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define VERBOSE_ONLY if constexpr (true)
#else
#define LOG(fmt, ...) \
  do {                \
  } while (0)
#define VERBOSE_ONLY if constexpr (false)
#endif

// Debugging macros
#ifdef DEBUG
#define DEBUG_ONLY if constexpr (true)
#else
#define DEBUG_ONLY if constexpr (false)
#endif
