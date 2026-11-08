"""RADV keeps the driver's optimizer for task/mesh shaders, and publishes the task payload with output memory.

Unoptimized task/mesh programs and a payload barrier that only names workgroup memory hang the gfx ring on
RADV (Steam Deck OLED, and the same timeouts on other RADV GPUs). AMD's proprietary compiler still builds
those programs unoptimized. The Deck profile is applied only when the machine is a Deck or Van Gogh.
python -I -B tests/test_radv_mesh_deck.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAW = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
MESH = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp").read_text(encoding="utf-8")
PROFILE = (ROOT / "gpu/vk/eden/shader_recompiler/profile.h").read_text(encoding="utf-8")
SHADER = (ROOT / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
MAIN = (ROOT / "host/hos_main.c").read_text(encoding="utf-8")


def main():
    if "radv_driver()" not in DRAW or "RX_VK_MESH_OPT=0" not in DRAW:
        raise SystemExit("vk_draw.cpp: RADV must keep the mesh optimizer unless RX_VK_MESH_OPT=0")
    if "amd && !radv" not in DRAW:
        raise SystemExit("vk_draw.cpp: the unoptimized mesh path must be proprietary AMD only")
    if "mesh_payload_output_memory" not in PROFILE or "mesh_payload_output_memory" not in SHADER:
        raise SystemExit("the payload barrier flag is not on the shader profile")
    barrier = MESH[MESH.index("void Barrier("):MESH.index("Id LocalInvocation")]
    if "0x1000u" not in barrier or "mesh_payload_output_memory" not in barrier:
        raise SystemExit("task/mesh barriers do not add output memory when the profile asks")
    if "WorkgroupMemory" not in barrier:
        raise SystemExit("the workgroup barrier went away")
    if "valve_deck" not in MAIN or "HOOKSHOT_OCCLUSION_OFF" not in MAIN or "RX_VK_FPS_CAP" not in MAIN:
        raise SystemExit("hos_main.c: Deck profile is missing")
    if "RX_VK_DECK=0" not in MAIN:
        raise SystemExit("hos_main.c: RX_VK_DECK=0 must turn the profile off")
    print("radv mesh optimizer, payload barrier and deck profile ok")


if __name__ == "__main__":
    main()
