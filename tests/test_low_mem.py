"""Deck / low-RAM limits stay off a large machine unless asked, and each one has an off switch.

python -I -B tests/test_low_mem.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def need(path, *parts):
    text = (ROOT / path).read_text(encoding="utf-8")
    for part in parts:
        if part not in text:
            raise SystemExit(f"{path}: missing {part}")
    return text


def main():
    need("gpu/vk/vk_main.cpp", "bool low_mem_on", "RX_VK_LOW_MEM", "MemTotal:", "13ull * 1024 * 1024",
         "C++ allocation failed", "preparing shaders before the game", "jobs_start")
    need("gpu/vk/vk_internal.h", "bool low_mem_on(const char* off_switch);")
    need("gpu/vk/vk_texture.cpp", "RX_VK_LOW_TEX", "896ull << 20", "RX_VK_LOW_STAGING", "RX_VK_EVICT_MIN_MS")
    need("gpu/vk/vk_shader.cpp", "POSIX_FADV_DONTNEED", "index only", "spirv_env")
    need("gpu/vk/vk_draw.cpp", "RX_VK_PRECOMPILE_THREADS", "RX_VK_PRECOMPILE_FREE_MB", "shader prep")
    need("gpu/vk/vk_jobs.cpp", "RX_VK_LOW_COMPILE", "1 background worker")
    mem = need("host/hos_mem.c", "4608ULL", "RX_VK_LOW_HEAP", "9809ULL")
    if "RX_VK_DECK" in mem or "MemTotal:" in mem:
        raise SystemExit("host/hos_mem.c: the 4608 MB heap must be RX_VK_LOW_HEAP=1 only")
    need("host/hos_main.c", "RX_VK_LOW_MEM=0", "HOS_GUEST_MEMFD", "9809 MB", "RX_VK_LOW_HEAP=1")
    need("host/posix/win32_posix.c", "_exit(128 + sig)")
    need("gpu/vk/vk_replay.cpp", "RX_VK_BATCH_HOLD_FIRST", "the first submit is sent now")
    need("gpu/vk/vk_memory.cpp", "udmabuf", "RX_VK_NO_DMABUF")
    stamp = (ROOT / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
    block = stamp[stamp.find("spirv_env"):stamp.find("spirv_env") + 700]
    for banned in ("RX_VK_DECK", "RX_VK_LOW_MEM", "RX_VK_LOW_HEAP", "RX_VK_MESH_OPT"):
        if banned in block:
            raise SystemExit(f"program-cache stamp must not include {banned}")
    print("low-memory switches ok")


if __name__ == "__main__":
    main()
