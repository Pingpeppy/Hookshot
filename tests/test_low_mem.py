"""Deck / low-RAM limits stay off a large machine unless RX_VK_LOW_MEM=1, and each one has an off switch.

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
    need("gpu/vk/vk_main.cpp", "bool low_mem_mode()", "RX_VK_LOW_MEM", "MemTotal:", "13ull * 1024 * 1024",
         "C++ allocation failed")
    need("gpu/vk/vk_internal.h", "bool low_mem_on(const char* off_switch);")
    need("gpu/vk/vk_texture.cpp", "RX_VK_LOW_TEX", "1280ull << 20", "RX_VK_LOW_STAGING",
         "MemAvailable:", "peak", "textures")
    need("gpu/vk/vk_memory.cpp", "RX_VK_LOW_CAPTURES", "RX_VK_CAPTURE_KEEP", "RX_VK_LOW_IMPORT", "RX_VK_LOW_MIRROR")
    need("gpu/vk/vk_shader.cpp", "RX_VK_LOW_PROGRAMS", "POSIX_MADV_DONTNEED", "disk_map")
    need("gpu/vk/vk_draw.cpp", "RX_VK_LOW_PIPELINES", "mapped and released")
    need("gpu/vk/vk_replay.cpp", "present returned")
    need("host/posix/display_sdl.c", "SDL_WINDOWEVENT_CLOSE", "SDL_QUIT", "MemAvailable", "not a GPU reset", "SIGTERM")
    need("host/display.c", "WM_CLOSE")
    need("host/hos_main.c", "RX_VK_LOW_MEM=0")
    print("low-memory switches ok")


if __name__ == "__main__":
    main()
