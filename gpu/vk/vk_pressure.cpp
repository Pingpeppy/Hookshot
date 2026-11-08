// Deck memory pressure (RX_VK_MEM_WATCH, on with low-memory mode). Linux reads MemAvailable at most every 250 ms.
#include "vk_internal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mkwvk {

bool mem_watch_on() {
    if (const char* e = getenv("RX_VK_MEM_WATCH")) return e[0] != '0';
    return low_mem_on(nullptr);
}

namespace {
uint64_t env_mb(const char* name, uint64_t def) {
    const char* e = getenv(name);
    return e && e[0] ? (uint64_t)std::atoll(e) << 20 : def;
}
bool read_meminfo(uint64_t* avail) {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return false;
    char line[128];
    *avail = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu", avail) == 1) { *avail <<= 10; break; }
    std::fclose(f);
    return *avail != 0;
}
} // namespace

uint64_t deck_gpu_cap() {
    if (const char* e = getenv("RX_VK_DECK_GPU_MB")) return e[0] ? (uint64_t)std::atoll(e) << 20 : ~0ull;
    static uint64_t cap = 0;
    if (cap) return cap;
    for (int i = 0; i < 8 && !cap; i++) {
        char path[160];
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_total", i);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        unsigned long long bytes = 0;
        if (std::fscanf(f, "%llu", &bytes) == 1 && bytes) cap = (uint64_t)bytes + (1024ull << 20);
        std::fclose(f);
    }
    if (!cap) cap = ~0ull >> 1;
    return cap;
}

MemPressure mem_pressure() {
    MemPressure p{0, ~0ull >> 1, 0};
#ifndef _WIN32
    if (!mem_watch_on()) return p;
    static int level = 0;
    static uint64_t cached = ~0ull >> 1;
    static auto at = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (at.time_since_epoch().count() == 0 || now - at >= std::chrono::milliseconds(250)) {
        uint64_t avail = 0;
        if (read_meminfo(&avail)) cached = avail;
        at = now;
    }
    const uint64_t low = env_mb("RX_VK_MEM_LOW_MB", 1536ull << 20);
    const uint64_t crit = env_mb("RX_VK_MEM_CRIT_MB", 768ull << 20);
    const uint64_t hyst = 512ull << 20;
    int next = level;
    if (cached < crit) next = 2;
    else if (level >= 2) {
        if (cached > crit + hyst) next = cached < low ? 1 : (cached > low + hyst ? 0 : 1);
    } else if (cached < low) next = 1;
    else if (level == 1 && cached > low + hyst) next = 0;
    level = next;
    p.level = level;
    p.avail = cached;
    const uint64_t thr = level >= 2 ? crit : low;
    p.deficit = level && cached < thr ? thr - cached : 0;
#else
    (void)0;
#endif
    return p;
}
int mem_pressure_level() { return mem_pressure().level; }

} // namespace mkwvk
