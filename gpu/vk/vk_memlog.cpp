// The Deck memory line (RX_VK_DECK_MEMLOG, on with low-memory mode). The present thread fills DeckMemSnap.
// Reading /proc and sysfs waits on a worker so a swap storm does not stall the frame.
#include "vk_internal.h"
#include "vk_jobs.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <malloc.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mkwvk {

bool deck_memlog_on() {
    if (const char* e = getenv("RX_VK_DECK_MEMLOG")) return e[0] != '0';
    return low_mem_on(nullptr);
}
int mem_pressure_level() { return 0; }

namespace {
uint64_t kb_of(const char* text, const char* key) {
    const char* p = std::strstr(text, key);
    if (!p) return 0;
    return std::strtoull(p + std::strlen(key), nullptr, 10);
}
bool read_file(const char* path, char* buf, size_t n) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    const size_t got = std::fread(buf, 1, n - 1, f);
    std::fclose(f);
    buf[got] = 0;
    return got > 0;
}
uint64_t sysfs_u64(const char* path) {
    char buf[64];
    if (!read_file(path, buf, sizeof buf)) return 0;
    return std::strtoull(buf, nullptr, 10);
}
void amdgpu_totals(uint64_t* vram_used, uint64_t* vram_total, uint64_t* gtt_used, uint64_t* gtt_total) {
    *vram_used = *vram_total = *gtt_used = *gtt_total = 0;
    for (int i = 0; i < 8; i++) {
        char path[160];
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_total", i);
        const uint64_t total = sysfs_u64(path);
        if (!total) continue;
        *vram_total = total;
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_used", i);
        *vram_used = sysfs_u64(path);
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_gtt_total", i);
        *gtt_total = sysfs_u64(path);
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_gtt_used", i);
        *gtt_used = sysfs_u64(path);
        return;
    }
}
void proc_fd_mem(uint64_t* memfd, uint64_t* proc_vram, uint64_t* proc_gtt) {
    *memfd = *proc_vram = *proc_gtt = 0;
    DIR* d = opendir("/proc/self/fd");
    if (!d) return;
    char link[256], path[64];
    while (dirent* ent = readdir(d)) {
        if (ent->d_name[0] == '.') continue;
        std::snprintf(path, sizeof path, "/proc/self/fd/%s", ent->d_name);
        const ssize_t n = readlink(path, link, sizeof link - 1);
        if (n <= 0) continue;
        link[n] = 0;
        if (!*memfd && std::strncmp(link, "/memfd:hookshot-guest", 21) == 0) {
            struct stat st {};
            int fd = std::atoi(ent->d_name);
            if (fd > 0 && fstat(fd, &st) == 0) *memfd = (uint64_t)st.st_blocks * 512;
        }
        if (!*proc_vram && std::strncmp(link, "/dev/dri/renderD", 16) == 0) {
            std::snprintf(path, sizeof path, "/proc/self/fdinfo/%s", ent->d_name);
            char info[4096];
            if (!read_file(path, info, sizeof info)) continue;
            // (amdgpu fdinfo reports these in KiB)
            *proc_vram = kb_of(info, "drm-memory-vram:") << 10;
            *proc_gtt = kb_of(info, "drm-memory-gtt:") << 10;
        }
    }
    closedir(d);
}
uint64_t mb(uint64_t b) { return b >> 20; }
} // namespace

void deck_memlog_emit(DeckMemSnap snap) {
    jobs_push([snap] {
        char status[2048] = {}, meminfo[4096] = {};
        read_file("/proc/self/status", status, sizeof status);
        read_file("/proc/meminfo", meminfo, sizeof meminfo);
        const uint64_t rss = kb_of(status, "VmRSS:") << 10;
        const uint64_t anon = kb_of(status, "RssAnon:") << 10;
        const uint64_t shmem = kb_of(status, "RssShmem:") << 10;
        const uint64_t file = kb_of(status, "RssFile:") << 10;
        const uint64_t proc_swap = kb_of(status, "VmSwap:") << 10;
        const uint64_t avail = kb_of(meminfo, "MemAvailable:") << 10;
        const uint64_t free = kb_of(meminfo, "MemFree:") << 10;
        const uint64_t swap_total = kb_of(meminfo, "SwapTotal:") << 10;
        const uint64_t swap_free = kb_of(meminfo, "SwapFree:") << 10;
        const uint64_t swap_used = swap_total > swap_free ? swap_total - swap_free : 0;
        uint64_t memfd = 0, proc_vram = 0, proc_gtt = 0, vram_u = 0, vram_t = 0, gtt_u = 0, gtt_t = 0;
        proc_fd_mem(&memfd, &proc_vram, &proc_gtt);
        amdgpu_totals(&vram_u, &vram_t, &gtt_u, &gtt_t);
        const uint64_t nonvma = snap.dev_used > snap.dev_blocks ? snap.dev_used - snap.dev_blocks : 0;
        static DeckMemSnap prev {};
        static double prev_t = 0;
        const double dt = prev_t > 0 ? std::max(snap.seconds - prev_t, 0.001) : 0;
        auto per_min = [&](uint64_t now, uint64_t was) -> long long {
            if (!dt) return 0;
            return (long long)((double)((int64_t)now - (int64_t)was) / dt * 60.0 / (1024.0 * 1024.0));
        };
        const long long d_heap = per_min(snap.dev_used, prev.dev_used);
        const long long d_tex = per_min(snap.tex, prev.tex);
        const long long d_memfd = per_min(memfd, 0); // filled below once prev memfd is kept
        static uint64_t prev_memfd = 0, prev_swap = 0;
        const long long d_memfd_b = dt ? (long long)((double)((int64_t)memfd - (int64_t)prev_memfd) / dt * 60.0 / (1024.0 * 1024.0)) : 0;
        const long long d_swap = dt ? (long long)((double)((int64_t)swap_used - (int64_t)prev_swap) / dt * 60.0 / (1024.0 * 1024.0)) : 0;
        (void)d_memfd;
        uint64_t malloc_free = 0;
#if defined(__linux__) && defined(__GLIBC__)
        malloc_free = mallinfo2().fordblks;
#endif
        const uint64_t streams = mem_tag_bytes(MemTag::Stream);
        const uint64_t bc = mem_tag_bytes(MemTag::BcScratch);
        uint64_t recordings = 0;
        if (auto* fn = (uint64_t (*)())dlsym(RTLD_DEFAULT, "rx_nvn_copy_bytes")) recordings = fn();
        logf("vk: memlog t=%.1fs present=%llu sys: avail=%llu free=%llu swap_used=%llu | proc: rss=%llu anon=%llu shmem=%llu "
             "file=%llu swap=%llu | guest: memfd=%llu imported=%llu | amdgpu: vram=%llu/%llu gtt=%llu/%llu proc_vram=%llu proc_gtt=%llu",
             snap.seconds, (unsigned long long)snap.present, (unsigned long long)mb(avail), (unsigned long long)mb(free),
             (unsigned long long)mb(swap_used), (unsigned long long)mb(rss), (unsigned long long)mb(anon), (unsigned long long)mb(shmem),
             (unsigned long long)mb(file), (unsigned long long)mb(proc_swap), (unsigned long long)mb(memfd),
             (unsigned long long)mb(snap.imports), (unsigned long long)mb(vram_u), (unsigned long long)mb(vram_t),
             (unsigned long long)mb(gtt_u), (unsigned long long)mb(gtt_t), (unsigned long long)mb(proc_vram),
             (unsigned long long)mb(proc_gtt));
        logf("vk: memlog gpu: heap1 used=%llu blocks=%llu alloc=%llu free_in_blocks=%llu nonvma=%llu | heap0 used=%llu blocks=%llu "
             "alloc=%llu imports=%llu | tex=%llu (n=%llu recent8=%llu astc=%llu bc3=%llu rt=%llu other=%llu budget=%llu)",
             (unsigned long long)mb(snap.dev_used), (unsigned long long)mb(snap.dev_blocks), (unsigned long long)mb(snap.dev_alloc),
             (unsigned long long)mb(snap.dev_free), (unsigned long long)mb(nonvma), (unsigned long long)mb(snap.sys_used),
             (unsigned long long)mb(snap.sys_blocks), (unsigned long long)mb(snap.sys_alloc), (unsigned long long)mb(snap.imports),
             (unsigned long long)mb(snap.tex), (unsigned long long)snap.tex_n, (unsigned long long)mb(snap.recent8),
             (unsigned long long)mb(snap.astc), (unsigned long long)mb(snap.bc3), (unsigned long long)mb(snap.rt),
             (unsigned long long)mb(snap.other), (unsigned long long)mb(snap.budget));
        logf("vk: memlog host: captures=%llu/%llu spares=%llu/%llu shadow=%llu staging=%llu streams=%llu bc_scratch=%llu "
             "recordings=%llu pipelines=%llu programs=%llu malloc_free=%llu pressure=L%d (MB; Δ/min: heap1 %+lld tex %+lld memfd %+lld swap %+lld)",
             (unsigned long long)snap.cap_n, (unsigned long long)mb(snap.cap_b), (unsigned long long)snap.spare_n,
             (unsigned long long)mb(snap.spare_b), (unsigned long long)mb(snap.shadow), (unsigned long long)mb(snap.staging),
             (unsigned long long)mb(streams), (unsigned long long)mb(bc), (unsigned long long)mb(recordings),
             (unsigned long long)snap.pipelines, (unsigned long long)snap.programs, (unsigned long long)mb(malloc_free),
             snap.pressure, d_heap, d_tex, d_memfd_b, d_swap);
        char kinds[640];
        mem_tag_format(kinds, sizeof kinds);
        logf("%s", kinds);
        prev = snap;
        prev_t = snap.seconds;
        prev_memfd = memfd;
        prev_swap = swap_used;
    });
}

} // namespace mkwvk
