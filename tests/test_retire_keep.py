"""Behaviour of keep_image_that_would_clear. A desktop retire must not gain the Deck keeps,
and a scene purge must not destroy an image that would come back cleared.
"""
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "gpu/vk/vk_retire.h"

SOURCE = r"""
#include <stdio.h>
#include "vk_retire.h"
static int fails;
static void expect(const char* name, bool got, bool want) {
    if (got == want) return;
    fprintf(stderr, "%s: got %d want %d\n", name, (int)got, (int)want);
    fails++;
}
int main() {
    expect("desktop gpu", keep_image_that_would_clear(false, false, false, true, false, false, false, false), false);
    expect("desktop bc", keep_image_that_would_clear(false, false, false, false, false, false, false, true), false);
    expect("ws bc", keep_image_that_would_clear(true, false, false, false, false, false, false, true), true);
    expect("ws gpu no writeback", keep_image_that_would_clear(true, false, false, true, false, false, false, false), true);
    expect("ws gpu written back", keep_image_that_would_clear(true, false, false, true, false, false, true, false), false);
    expect("ws gpu evict rendered", keep_image_that_would_clear(true, false, false, true, false, true, false, false), false);
    expect("scene bc", keep_image_that_would_clear(false, true, false, false, false, false, false, true), true);
    expect("scene gpu", keep_image_that_would_clear(true, true, false, true, false, false, false, false), true);
    expect("scene overwritten bc", keep_image_that_would_clear(true, true, true, false, false, false, false, true), false);
    expect("pressure is not an input", keep_image_that_would_clear(true, false, false, false, false, false, false, true), true);
    return fails ? 1 : 0;
}
"""


def main():
    text = HEADER.read_text(encoding="utf-8")
    if "ws || scene" not in text:
        raise SystemExit("vk_retire.h: the keep must be the working-set pass or a scene purge, not desktop")
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "retire_keep.cpp"
        src.write_text(SOURCE, encoding="utf-8")
        out = pathlib.Path(d) / "retire_keep"
        cmd = ["clang++", "-std=c++20", "-I", str(ROOT / "gpu/vk"), str(src), "-o", str(out)]
        subprocess.check_call(cmd)
        subprocess.check_call([str(out)])
    print("retire keep ok")


if __name__ == "__main__":
    main()
