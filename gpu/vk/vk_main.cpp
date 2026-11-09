// The Vulkan back end's start: vulkan-1.dll, the instance, the device (Vulkan 1.3 and the extensions the NVN replay
// wants where the GPU has them: mesh shaders, host memory import, graphics pipeline libraries), the window's
// swapchain, and the rx_* table (vk_api.h). The NVN replay itself is vk_replay.cpp.
#include "vk_paths.h"
#include "vk_loader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "vk_api.h"
#include "vk_draw.h"
#include "vk_internal.h"
#include "gpu_choice.h"
#include "vk_jobs.h"
#include "vk_menu.h"

namespace vkx {
#define VKX_DEFINE(f) PFN_##f f;
VKX_GLOBAL(VKX_DEFINE)
VKX_INSTANCE(VKX_DEFINE)
VKX_DEVICE(VKX_DEFINE)
VKX_DEVICE2(VKX_DEFINE)
VKX_DEVICE3(VKX_DEFINE)
#undef VKX_DEFINE

static PFN_vkGetInstanceProcAddr get_instance_proc;

bool load_global() {
    HMODULE m = LoadLibraryA("vulkan-1.dll");   // (linux: libvulkan.so.1, host/posix/win32_posix.c)
    if (!m) return false;
    get_instance_proc = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(m, "vkGetInstanceProcAddr");
    if (!get_instance_proc) return false;
#define VKX_LOAD(f) f = (PFN_##f)get_instance_proc(nullptr, #f);
    VKX_GLOBAL(VKX_LOAD)
#undef VKX_LOAD
    return vkCreateInstance != nullptr;
}
PFN_vkGetInstanceProcAddr instance_proc_addr() { return get_instance_proc; }
void load_instance(VkInstance instance) {
#define VKX_LOAD(f) f = (PFN_##f)get_instance_proc(instance, #f);
    VKX_INSTANCE(VKX_LOAD)
#undef VKX_LOAD
}
void load_device(VkDevice device) {
#define VKX_LOAD(f) f = (PFN_##f)vkGetDeviceProcAddr(device, #f);
    VKX_DEVICE(VKX_LOAD)
    VKX_DEVICE2(VKX_LOAD)
    VKX_DEVICE3(VKX_LOAD)
#undef VKX_LOAD
}
} // namespace vkx

namespace mkwvk {

std::vector<std::string> gpu_device_names;
const MkwVkHost* host;
int64_t prof_ns[PROF_COUNT];
Device dev;
PFN_vkGetInstanceProcAddr instance_proc() { return vkx::instance_proc_addr(); }

void logf(const char* fmt, ...) {
    char b[1024];
    std::va_list a;
    va_start(a, fmt);
    std::vsnprintf(b, sizeof b, fmt, a);
    va_end(a);
    if (host) host->log(b);
}

namespace {

bool has_extension(const std::vector<VkExtensionProperties>& all, const char* name) {
    for (const auto& e : all)
        if (!std::strcmp(e.extensionName, name)) return true;
    return false;
}

bool create_instance() {
    uint32_t version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&version);
    if (version < VK_API_VERSION_1_3) {
        logf("vk: the Vulkan loader is %u.%u: 1.3 is needed", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version));
        return false;
    }
#ifdef _WIN32
    const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    const uint32_t n_exts = 2;
#else
    // (linux: every window system's surface the loader has, for whichever one SDL's window is on: X11 or Wayland)
    const char* exts[4] = {VK_KHR_SURFACE_EXTENSION_NAME};
    uint32_t n_exts = 1;
    {
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> all(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, all.data());
        for (const char* want : {"VK_KHR_xlib_surface", "VK_KHR_xcb_surface", "VK_KHR_wayland_surface"})
            if (has_extension(all, want)) exts[n_exts++] = want;
    }
#endif
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "mkw-recomp";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = n_exts;
    ci.ppEnabledExtensionNames = exts;
    // RX_VK_VALIDATION=1: the Khronos validation layer (when installed)
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (getenv("RX_VK_VALIDATION")) ci.enabledLayerCount = 1, ci.ppEnabledLayerNames = &layer;
    if (vkCreateInstance(&ci, nullptr, &dev.instance) != VK_SUCCESS) {
        logf("vk: vkCreateInstance failed");
        return false;
    }
    load_instance(dev.instance);
    return true;
}

bool pick_device() {
    menu_init();
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(dev.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> all(n);
    vkEnumeratePhysicalDevices(dev.instance, &n, all.data());

    std::vector<uint32_t> valid_gpus;
    gpu_device_names.clear();
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(all[i], &p);
        logf("vk: device %u: %s, Vulkan %u.%u, driver %#x, type %d", i, p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
             VK_API_VERSION_MINOR(p.apiVersion), p.driverVersion, (int)p.deviceType);
        if (p.apiVersion >= VK_API_VERSION_1_3) {
            valid_gpus.push_back(i);
            gpu_device_names.push_back(p.deviceName);
        }
    }
    menu_set_gpu_count((int)valid_gpus.size());

    // (AMD laptops) With AMD graphics in the processor and an AMD card, Windows has two AMD drivers installed, one an
    // adapter, and each one's Vulkan driver lists every AMD GPU: the card shows up twice. It came first through the
    // processor graphics' driver, which the card's own update leaves as it was: on an RX 7700S that older driver
    // (amdvlk64.dll 32.0.11036.4002) crashed in its shader compiler (0xc0000409) on the game's compute pipelines while
    // the card's 32.0.32015.7029 built them. The same kind of GPU twice: the one with the newer driver wins.
    // RX_VK_DEVICE=<n> or opts.gpu_device: that device, as listed in the log
    int best = -1, best_score = -1;
    uint32_t best_driver = 0;
    int target_raw_idx = -1;
    if (const char* want = getenv("RX_VK_DEVICE")) {
        if (*want) target_raw_idx = atoi(want);
    } else if (const int k = gpu_choice::menu_device(opts.gpu_device, (int)valid_gpus.size()); k >= 0) {
        target_raw_idx = (int)valid_gpus[k];
    }

    for (uint32_t i : valid_gpus) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(all[i], &p);
        if (target_raw_idx >= 0) {
            if ((int)i == target_raw_idx) best = (int)i, best_score = 3;
            continue;
        }
        const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                          p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        if (score > best_score || (score == best_score && p.driverVersion > best_driver))
            best = (int)i, best_score = score, best_driver = p.driverVersion;
    }
    if (best < 0 && target_raw_idx >= 0) {
        logf("vk: preferred device %d not available or lacks Vulkan 1.3, falling back to auto selection", target_raw_idx);
        for (uint32_t i : valid_gpus) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(all[i], &p);
            const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                              p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
            if (score > best_score || (score == best_score && p.driverVersion > best_driver))
                best = (int)i, best_score = score, best_driver = p.driverVersion;
        }
    }
    if (best < 0) {
        logf("vk: no Vulkan 1.3 GPU");
        return false;
    }
    dev.physical = all[best];
    vkGetPhysicalDeviceProperties(dev.physical, &dev.props);
    vkGetPhysicalDeviceMemoryProperties(dev.physical, &dev.memory);
    return true;
}

bool create_device() {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, exts.data());
    std::vector<const char*> want = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    auto optional = [&](const char* name, bool& flag) {
        flag = has_extension(exts, name);
        if (flag) want.push_back(name);
    };
    optional(VK_EXT_MESH_SHADER_EXTENSION_NAME, dev.mesh_shader);
    optional(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, dev.external_memory_host);
    optional(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, dev.external_memory_fd);   // (guest memory as udmabufs: vk_memory.cpp import_dma_buf)
    if (dev.external_memory_fd) optional(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME, dev.dma_buf_import);
    optional(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, dev.pipeline_library);
    if (dev.pipeline_library) optional(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, dev.graphics_pipeline_library);
    optional(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, dev.custom_border_color);
    optional(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, dev.dynamic_state3);
    optional(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, dev.depth_clip_control);
    optional(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, dev.null_descriptor);
    optional(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, dev.index_uint8);
    optional(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, dev.push_descriptor);
    optional(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, dev.conditional_rendering);
    optional(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, dev.provoking_vertex);
    optional(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, dev.vertex_divisor);
    optional(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME, dev.stencil_export);
    optional(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, dev.memory_budget);
    optional(VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME, dev.view_2d_of_3d);
    // (hookshot) on by default: with a marker before each draw and dispatch the device has not been lost (five
    // runs); without them it was, every run, shortly after OoT's first gameplay frames (nvlddmkm event 153; the
    // extension alone, RX_VK_CHECKPOINTS=ext, did not keep it). The race they hide is not found yet.
    // RX_VK_CHECKPOINTS=0: off
    if (!getenv("RX_VK_CHECKPOINTS") || strcmp(getenv("RX_VK_CHECKPOINTS"), "0") != 0)
        optional(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, dev.checkpoints);
    // (asked for only: a marker written twice a draw, where the NV ones are not there to hide a race)
    if (const char* c = getenv("RX_VK_CHECKPOINTS"); !dev.checkpoints && c && strcmp(c, "0") != 0 && strcmp(c, "ext") != 0)
        optional(VK_AMD_BUFFER_MARKER_EXTENSION_NAME, dev.buffer_marker);
    optional(VK_EXT_DEVICE_FAULT_EXTENSION_NAME, dev.device_fault);
    optional(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, dev.coop_matrix);

    // the queue: one family for graphics, compute and transfer
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &n, families.data());
    dev.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < n && dev.queue_family == UINT32_MAX; i++)
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            dev.queue_family = i;
    if (dev.queue_family == UINT32_MAX) return false;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = dev.queue_family;
    q.queueCount = 1;
    q.pQueuePriorities = &priority;

    // features: what the hardware has of what the replay uses (all of them on the GPUs MKW-class hardware maps to)
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT robust2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT u8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f.pNext = &f11, f11.pNext = &f12, f12.pNext = &f13;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT border{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    VkPhysicalDeviceProvokingVertexFeaturesEXT pv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    void** tail = &f13.pNext;
    if (dev.mesh_shader) *tail = &mesh, tail = &mesh.pNext;
    if (dev.null_descriptor) *tail = &robust2, tail = &robust2.pNext;
    if (dev.index_uint8) *tail = &u8, tail = &u8.pNext;
    if (dev.dynamic_state3) *tail = &eds3, tail = &eds3.pNext;
    if (dev.custom_border_color) *tail = &border, tail = &border.pNext;
    if (dev.depth_clip_control) *tail = &clip, tail = &clip.pNext;
    if (dev.conditional_rendering) *tail = &cond, tail = &cond.pNext;
    if (dev.provoking_vertex) *tail = &pv, tail = &pv.pNext;
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT vdiv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT};
    if (dev.vertex_divisor) *tail = &vdiv, tail = &vdiv.pNext;
    if (dev.graphics_pipeline_library) *tail = &gpl, tail = &gpl.pNext;
    VkPhysicalDeviceFaultFeaturesEXT fault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    if (dev.device_fault) *tail = &fault, tail = &fault.pNext;
    VkPhysicalDeviceImage2DViewOf3DFeaturesEXT v2d3d{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT};
    if (dev.view_2d_of_3d) *tail = &v2d3d, tail = &v2d3d.pNext;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    if (dev.coop_matrix) *tail = &coop, tail = &coop.pNext;
    vkGetPhysicalDeviceFeatures2(dev.physical, &f);
    dev.view_2d_of_3d = dev.view_2d_of_3d && v2d3d.sampler2DViewOf3D;
    coop.cooperativeMatrixRobustBufferAccess = VK_FALSE;
    dev.coop_matrix = dev.coop_matrix && coop.cooperativeMatrix &&
                      f12.vulkanMemoryModel && f12.vulkanMemoryModelDeviceScope;
    // (bounds-checked buffer access: an out-of-range access dropped instead of faulting the device. On by default: the GPU
    // culling pass 53caac5f faulted the device while a save loaded, about one start in three, and the scene's lighting read garbage
    // for single frames (the black flashes); with it, neither in A/B runs at the same GPU time. Lichii's OoT branch keeps it on too.
    // The 'Robust buffers' row of hookshot_options.ini chooses (0 Auto: on, on every vendor; 1 Off; 2 On); RX_VK_ROBUST=0 or 1
    // over it. Off is null descriptors only: some NVIDIA cards ran slower with it on, issues 394, 418, 419)
    static const bool robust = []() -> bool {
        if (const char* e = getenv("RX_VK_ROBUST")) return atoi(e) != 0;
        char exe[MAX_PATH];
        const DWORD n = game_module_path(exe, MAX_PATH);   // (the game folder, not bin beside the launcher: vk_paths.h)
        std::string ini(exe, n);
        ini = ini.substr(0, ini.find_last_of("\\/") + 1) + "hookshot_options.ini";
        int v = 0;   // (Auto)
        if (FILE* f = fopen(ini.c_str(), "r")) {
            char line[80];
            while (fgets(line, sizeof line, f))
                if (!strncmp(line, "robust_buffers=", 15)) {
                    v = atoi(line + 15);
                    break;
                }
            fclose(f);   // (once: the PR closed it twice when the row said Auto)
        }
        return v != 1;   // (Auto and On: on)
    }();
    const bool robust_ok = robust && robust2.robustBufferAccess2 && f.features.robustBufferAccess;
    robust2.robustBufferAccess2 = robust_ok ? VK_TRUE : VK_FALSE;
    robust2.robustImageAccess2 = VK_FALSE;
    dev.null_descriptor = dev.null_descriptor && robust2.nullDescriptor;
    pv.transformFeedbackPreservesProvokingVertex = VK_FALSE;
    dev.provoking_vertex = dev.provoking_vertex && pv.provokingVertexLast;
    dev.conditional_rendering = dev.conditional_rendering && cond.conditionalRendering;
    dev.custom_border_color = dev.custom_border_color && border.customBorderColors && border.customBorderColorWithoutFormat;
    dev.depth_clip_control = dev.depth_clip_control && clip.depthClipControl;
    dev.graphics_pipeline_library = dev.graphics_pipeline_library && gpl.graphicsPipelineLibrary;
    if (dev.graphics_pipeline_library) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT lp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &lp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        dev.graphics_pipeline_library = lp.graphicsPipelineLibraryFastLinking != VK_FALSE;
    }
    dev.eds3_polygon = dev.dynamic_state3 && eds3.extendedDynamicState3PolygonMode;
    dev.eds3_clamp = dev.dynamic_state3 && eds3.extendedDynamicState3DepthClampEnable;
    dev.eds3_blend = dev.dynamic_state3 && eds3.extendedDynamicState3ColorBlendEnable &&
                     eds3.extendedDynamicState3ColorBlendEquation && eds3.extendedDynamicState3ColorWriteMask;
    dev.eds3_a2c = dev.dynamic_state3 && eds3.extendedDynamicState3AlphaToCoverageEnable;
    dev.vk12 = f12;
    dev.vk13 = f13;
    {   // (hookshot, after TotK Recomp's vk_caps.c) NVN's programs are written for 32 lanes: the translator is told 32, and the lane
        // shuffles of OoT's networks and the lane masks of its mesh programs depend on it. NVIDIA's GPUs have no other size; AMD's
        // default to 64 where they like, and the networks' output came out as coloured noise (Hookshot issues 10 and 14)
        VkPhysicalDeviceSubgroupSizeControlProperties sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        sg.pNext = &sc;
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        const bool pin = f13.subgroupSizeControl && sc.minSubgroupSize <= 32 && sc.maxSubgroupSize >= 32 && !getenv("RX_VK_NO_SUBGROUP_PIN");
        dev.sg32_stages = pin ? sc.requiredSubgroupSizeStages : 0;
        logf("vk: subgroups: default %u lanes, %u-%u, pinned to 32 on stages %#x", sg.subgroupSize, sc.minSubgroupSize, sc.maxSubgroupSize,
             (unsigned)dev.sg32_stages);
    }
    if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore) {
        logf("vk: the GPU lacks dynamic rendering, synchronization2 or timeline semaphores");
        return false;
    }
    mesh.multiviewMeshShader = mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;   // (not used)
    dev.mesh_shader = dev.mesh_shader && mesh.meshShader && mesh.taskShader;
    // RX_VK_MESH_FALLBACK=1: the issue 640 compute path even when the device has mesh shaders (a backup when the
    // driver's mesh compiler faults). Off unless that variable is set. Upstream reverted that fallback; this branch keeps it.
    const bool force_mesh_fallback = [] {
        const char* e = getenv("RX_VK_MESH_FALLBACK");
        return e && e[0] && std::atoi(e) != 0;
    }();
    if (force_mesh_fallback) dev.mesh_shader = false;
    dev.mesh_fallback = !dev.mesh_shader && f12.bufferDeviceAddress && !getenv("RX_VK_NO_MESH_FALLBACK");
    if (dev.mesh_fallback && force_mesh_fallback)
        logf("vk: task and mesh programs run as compute shaders (RX_VK_MESH_FALLBACK=1; the device has mesh shaders)");
    else if (dev.mesh_fallback)
        logf("vk: no mesh shaders: task and mesh programs run as compute shaders (issue 640 fallback)");
    else if (force_mesh_fallback)
        logf("vk: RX_VK_MESH_FALLBACK=1, but this GPU has no buffer device addresses, so mesh programs are left out");
    if (robust) logf("vk: robust buffer access %s", robust_ok ? "on" : "not supported");
    logf("vk: nullDescriptor %s", dev.null_descriptor ? "on" : "off");
    f.features.robustBufferAccess = robust_ok ? VK_TRUE : VK_FALSE;   // (off by default: the game's accesses stay in their buffers)
    dev.features = f.features;
    dev.sparse_binding = (families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) && f.features.sparseBinding;
    dev.sparse_images = dev.sparse_binding &&
                        f.features.sparseResidencyImage2D && !(getenv("RX_VK_SPARSE") && !atoi(getenv("RX_VK_SPARSE")));
    dev.sparse_buffers = (families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) && f.features.sparseBinding;
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext = &f;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &q;
    ci.enabledExtensionCount = (uint32_t)want.size();
    ci.ppEnabledExtensionNames = want.data();
    if (vkCreateDevice(dev.physical, &ci, nullptr, &dev.device) != VK_SUCCESS) {
        logf("vk: vkCreateDevice failed");
        return false;
    }


    load_device(dev.device);
    vkGetDeviceQueue(dev.device, dev.queue_family, 0, &dev.queue);
    if (dev.buffer_marker) buffer_markers_init();
    coopmat_init();
    if (dev.push_descriptor) {
        VkPhysicalDevicePushDescriptorPropertiesKHR pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &pp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        dev.max_push_descriptors = pp.maxPushDescriptors;
    }
    return true;
}

} // namespace

bool create_swapchain(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps;
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev.physical, dev.surface, &caps);
    if (result != VK_SUCCESS) {
        logf("vk: querying surface capabilities failed (VkResult %d)", (int)result);
        return false;
    }
    if (caps.currentExtent.width != UINT32_MAX) width = caps.currentExtent.width, height = caps.currentExtent.height;
    logf("vk: surface caps images %u..%u extent %ux%u..%ux%u usage %#x alpha %#x transform %#x",
         caps.minImageCount, caps.maxImageCount, caps.minImageExtent.width, caps.minImageExtent.height,
         caps.maxImageExtent.width, caps.maxImageExtent.height, caps.supportedUsageFlags,
         caps.supportedCompositeAlpha, caps.supportedTransforms);
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        logf("vk: surface does not support transfer-destination swapchain images");
        return false;
    }
    if (!width || !height) return false;
    uint32_t n = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, dev.surface, &n, nullptr);
    if (result != VK_SUCCESS || !n) {
        logf("vk: querying surface formats failed (VkResult %d, count %u)", (int)result, n);
        return false;
    }
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, dev.surface, &n, formats.data());
    VkSurfaceFormatKHR fmt = formats[0];
    for (const auto& f : formats)   // (an UNORM target: the game's window textures are presented as they are)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) fmt = f;
    // mailbox when the driver has it (latest frame, no tear). The game paces itself (nn_vi.c).
    n = 0;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical, dev.surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes;
    if (result == VK_SUCCESS && n) {
        modes.resize(n);
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical, dev.surface, &n, modes.data());
    }
    if (result != VK_SUCCESS || modes.empty())
        logf("vk: querying present modes failed (VkResult %d, count %u); trying mandatory FIFO mode", (int)result, n);
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    const bool force_fifo = getenv("RX_VK_FIFO") != nullptr;   // (as on a card or driver with vsync forced)
    if (!force_fifo) {
        for (auto m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
        // (Windows, no mailbox: AMD's driver has none, and FIFO held anything under 60 fps at 30.
        //  Linux: immediate on NVIDIA's X11 path turns vblank off for every output, so the other
        //  monitor and other programs tear. RX_VK_IMMEDIATE=1 still asks for it.)
        const bool want_immediate = getenv("RX_VK_IMMEDIATE") != nullptr;
#ifdef _WIN32
        if (want_immediate || mode == VK_PRESENT_MODE_FIFO_KHR)
#else
        if (want_immediate)
#endif
            for (auto m : modes)
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
    }
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = dev.surface;
    ci.minImageCount = std::max(caps.minImageCount, 4u);   // (hookshot: mailbox with frame interpolation: an image always free)
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = {width, height};
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // The renderer copies into swapchain images. Readback is optional and only available on surfaces that allow it.
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & ci.compositeAlpha)) {
        for (VkCompositeAlphaFlagBitsKHR alpha : {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                                                  VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                                                  VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            if (caps.supportedCompositeAlpha & alpha) { ci.compositeAlpha = alpha; break; }
    }
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = dev.swapchain;
    VkSwapchainKHR sc;
    result = vkCreateSwapchainKHR(dev.device, &ci, nullptr, &sc);
    if (result != VK_SUCCESS) {
        logf("vk: vkCreateSwapchainKHR failed (VkResult %d, extent %ux%u, format %d, usage %#x, mode %d, images %u)",
             (int)result, width, height, (int)fmt.format, ci.imageUsage, (int)mode, ci.minImageCount);
        return false;
    }
    if (vkGetSwapchainImagesKHR(dev.device, sc, &n, nullptr) != VK_SUCCESS || !n) {
        vkDestroySwapchainKHR(dev.device, sc, nullptr);
        return false;
    }
    std::vector<VkSemaphore> rendered(n, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& semaphore : rendered) {
        if (vkCreateSemaphore(dev.device, &si, nullptr, &semaphore) != VK_SUCCESS) {
            for (auto made : rendered)
                if (made) vkDestroySemaphore(dev.device, made, nullptr);
            vkDestroySwapchainKHR(dev.device, sc, nullptr);
            return false;
        }
    }
    if (dev.swapchain) {
        queue_flush_locked();
        vkDeviceWaitIdle(dev.device);
        for (auto semaphore : dev.swap_rendered) vkDestroySemaphore(dev.device, semaphore, nullptr);
        vkDestroySwapchainKHR(dev.device, dev.swapchain, nullptr);
    }
    dev.swapchain = sc;
    dev.swap_rendered = std::move(rendered);
    dev.swap_format = fmt.format;
    dev.swap_extent = {width, height};
    vkGetSwapchainImagesKHR(dev.device, sc, &n, nullptr);
    dev.swap_images.resize(n);
    vkGetSwapchainImagesKHR(dev.device, sc, &n, dev.swap_images.data());
    logf("vk: swapchain %ux%u, %u images, %s", width, height, n,
         mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo");
    return true;
}

namespace {

HWND main_window = nullptr;
#ifndef _WIN32
const MkwVkWindow* prep_window = nullptr;
#endif
// 5x7, bit 4 is the leftmost pixel. Lower case is drawn as upper case.
const uint8_t* splash_glyph(char c) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 32);
    static const uint8_t font[59][7] = {
        {}, {}, {}, {}, {},
        {0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13}, // %
        {}, {},
        {0x04, 0x08, 0x10, 0x10, 0x10, 0x08, 0x04}, // (
        {0x10, 0x08, 0x04, 0x04, 0x04, 0x08, 0x10}, // )
        {}, {}, {}, {}, {},
        {0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10}, // /
        {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // 0
        {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
        {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
        {0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E},
        {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
        {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
        {0x0E, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x0E},
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
        {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
        {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E}, // 9
        {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00}, // :
        {}, {}, {}, {}, {}, {},
        {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, // A
        {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},
        {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},
        {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E},
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F},
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},
        {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E},
        {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
        {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
        {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C},
        {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},
        {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},
        {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},
        {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11},
        {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
        {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
        {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},
        {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},
        {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},
        {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04},
        {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A},
        {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
        {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04},
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}, // Z
    };
    if ((unsigned)c < 32 || (unsigned)c > 90) return font[0];
    return font[c - 32];
}

struct Splash {
    bool ready = false;
    bool failed = false;
    bool busy = false;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
    uint32_t width = 0;
    int last_pct = -2;
    std::chrono::steady_clock::time_point last_draw{};
    std::chrono::steady_clock::time_point phase_t0{};
    size_t phase_total = 0;
};
Splash splash;

// The progress bar asks for a turn. The driver thread, between creates, sets splash_yield and waits
// briefly so the bar can present without submitting during a compile. splash_stop clears both.
std::atomic<int> splash_want{0};
std::atomic<int> splash_yield{0};

void splash_free_buffer() {
    if (splash.mem) {
        auto unmap = (PFN_vkUnmapMemory)vkGetDeviceProcAddr(dev.device, "vkUnmapMemory");
        if (unmap && splash.map) unmap(dev.device, splash.mem);
        vkFreeMemory(dev.device, splash.mem, nullptr);
    }
    if (splash.buf) vkDestroyBuffer(dev.device, splash.buf, nullptr);
    splash.buf = VK_NULL_HANDLE;
    splash.mem = VK_NULL_HANDLE;
    splash.map = nullptr;
    splash.width = 0;
}

void splash_stop() {
    // A short wait destroyed the fence while the first present was still in it (gamescope, a slow queue).
    if (splash.busy && splash.fence)
        vkWaitForFences(dev.device, 1, &splash.fence, VK_TRUE, UINT64_MAX);
    splash.busy = false;
    splash_free_buffer();
    if (splash.acquired) vkDestroySemaphore(dev.device, splash.acquired, nullptr);
    if (splash.fence) vkDestroyFence(dev.device, splash.fence, nullptr);
    if (splash.pool) vkDestroyCommandPool(dev.device, splash.pool, nullptr);
    splash.acquired = VK_NULL_HANDLE;
    splash.fence = VK_NULL_HANDLE;
    splash.pool = VK_NULL_HANDLE;
    splash.cmd = VK_NULL_HANDLE;
    splash.ready = false;
    // The driver thread yields only while this is set. Leaving it set after prep waits ~100 ms
    // every couple of seconds on later compiles.
    splash_want.store(0, std::memory_order_release);
    splash_yield.store(0, std::memory_order_release);
}

bool splash_init() {
    if (splash.ready || splash.failed) return splash.ready;
    if (dev.offscreen || !dev.swapchain || dev.swap_images.empty()) return false;
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = dev.queue_family;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (vkCreateCommandPool(dev.device, &pi, nullptr, &splash.pool) != VK_SUCCESS) {
        splash.failed = true;
        return false;
    }
    ai.commandPool = splash.pool;
    if (vkAllocateCommandBuffers(dev.device, &ai, &splash.cmd) != VK_SUCCESS ||
        vkCreateFence(dev.device, &fi, nullptr, &splash.fence) != VK_SUCCESS ||
        vkCreateSemaphore(dev.device, &si, nullptr, &splash.acquired) != VK_SUCCESS) {
        splash_stop();
        splash.failed = true;
        return false;
    }
    splash.ready = true;
    logf("vk: shader prep draws its progress on the window");
    return true;
}

bool splash_buffer(uint32_t width) {
    constexpr uint32_t kBand = 56;
    if (splash.map && splash.width == width) return true;
    splash_free_buffer();
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < dev.memory.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = dev.memory.memoryTypes[i].propertyFlags;
        if ((f & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) return false;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = (VkDeviceSize)width * kBand * 4;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(dev.device, &bi, nullptr, &splash.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev.device, splash.buf, &req);
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = req.size;
    ma.memoryTypeIndex = type;
    if ((req.memoryTypeBits & (1u << type)) == 0 ||
        vkAllocateMemory(dev.device, &ma, nullptr, &splash.mem) != VK_SUCCESS ||
        vkBindBufferMemory(dev.device, splash.buf, splash.mem, 0) != VK_SUCCESS ||
        vkMapMemory(dev.device, splash.mem, 0, req.size, 0, (void**)&splash.map) != VK_SUCCESS) {
        splash_free_buffer();
        return false;
    }
    splash.width = width;
    return true;
}

void splash_text(uint8_t* px, uint32_t width, uint32_t height, int scale, const char* text, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (!text || !*text || scale < 1) return;
    const int gw = 6 * scale;
    const int n = (int)strlen(text);
    int x = ((int)width - n * gw) / 2;
    const bool bgra = dev.swap_format == VK_FORMAT_B8G8R8A8_UNORM;
    for (int i = 0; i < n; i++, x += gw) {
        const uint8_t* rows = splash_glyph(text[i]);
        for (int row = 0; row < 7; row++)
            for (int col = 0; col < 5; col++) {
                if ((rows[row] & (0x10 >> col)) == 0) continue;
                for (int dy = 0; dy < scale; dy++)
                    for (int dx = 0; dx < scale; dx++) {
                        const int px_x = x + col * scale + dx, px_y = y + row * scale + dy;
                        if (px_x < 0 || px_y < 0 || px_x >= (int)width || px_y >= (int)height) continue;
                        uint8_t* p = px + ((size_t)px_y * width + px_x) * 4;
                        if (bgra) p[0] = b, p[1] = g, p[2] = r;
                        else p[0] = r, p[1] = g, p[2] = b;
                        p[3] = 255;
                    }
            }
    }
}

void splash_title(const char* text) {
    char title[160];
    snprintf(title, sizeof title, "Hookshot — %s", text);
#ifdef _WIN32
    if (main_window) SetWindowTextA(main_window, title);
#else
    if (prep_window && prep_window->version >= 2 && prep_window->set_title)
        prep_window->set_title(prep_window->window, title);
#endif
}

// Set while the RADV driver thread is inside an ICD call. A queue submit then waits on the device
// lock that call holds, and the wait loop (the stall watchdog, the progress log) never runs.
std::atomic<int> radv_on_driver{0};

// The acquire succeeded and this call will not present. Submit a wait on the acquire semaphore and
// present, so the semaphore can be used again and the swapchain image is not held.
bool splash_release(uint32_t index) {
    if (index >= dev.swap_images.size() || index >= dev.swap_rendered.size() || !dev.swap_rendered[index] || !splash.cmd)
        return false;
    vkResetCommandPool(dev.device, splash.pool, 0);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(splash.cmd, &cbbi) != VK_SUCCESS) return false;
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = dev.swap_images[index];
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(splash.cmd, &di);
    if (vkEndCommandBuffer(splash.cmd) != VK_SUCCESS) return false;
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = splash.acquired;
    wait.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = dev.swap_rendered[index];
    signal.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkCommandBufferSubmitInfo cbs{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbs.commandBuffer = splash.cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cbs;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &dev.swap_rendered[index];
    present.swapchainCount = 1;
    present.pSwapchains = &dev.swapchain;
    present.pImageIndices = &index;
    std::lock_guard qk(dev.present_mu);
    queue_flush_locked();
    if (vkQueueSubmit2(dev.queue, 1, &submit, splash.fence) != VK_SUCCESS) return false;
    vkQueuePresentKHR(dev.queue, &present);
    splash.busy = true;
    return true;
}

// A clear plus one small copy. No pipeline is built, so this does not join the shader compiles.
// Desktop release builds draw this too (precompile_wanted is a release build, or RX_VK_PRECOMPILE).
// On Windows the PeekMessageA below pumps this thread's queue so the window, created on this thread,
// paints during prep. Linux leaves window events to the display thread. The picture size is unchanged
// on desktop; the progress window is the new behaviour.
void splash_progress(size_t done, size_t total) {
    if (dev.offscreen || !dev.swapchain) return;
#ifdef _WIN32
    MSG msg;
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_PAINT && msg.hwnd) {
            ValidateRect(msg.hwnd, nullptr);
            continue;
        }
        if (msg.message == WM_ERASEBKGND) continue;
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
#endif
    char line[120];
    char eta[48] = "";
    const auto now = std::chrono::steady_clock::now();
    if (!total) {
        snprintf(line, sizeof line, "Preparing shaders");
    } else {
        if (splash.phase_total != total || done < 1) {
            if (splash.phase_total != total) splash.phase_t0 = now, splash.phase_total = total;
        }
        const size_t pct = done * 100 / total;
        snprintf(line, sizeof line, "Preparing shaders: %zu / %zu (%zu%%)", done, total, pct);
        if (done > 0 && done < total) {
            const double elapsed = std::chrono::duration<double>(now - splash.phase_t0).count();
            const double left = elapsed * (double)(total - done) / (double)done;
            if (elapsed >= 1.0 && left >= 1.0) {
                if (left >= 90.0) snprintf(eta, sizeof eta, "about %u min left", (unsigned)(left / 60.0 + 0.5));
                else snprintf(eta, sizeof eta, "about %u s left", (unsigned)(left + 0.5));
            }
        }
    }
    char both[160];
    snprintf(both, sizeof both, "%s%s%s", line, eta[0] ? " — " : "", eta);
    splash_title(both);
    splash_want.store(1, std::memory_order_release);
    // A submit during a compile blocks on the device lock that compile holds, and the precompile
    // wait (its log, its stall watchdog) stops with it. The driver thread yields between creates.
    const bool turn = splash_yield.load(std::memory_order_acquire) != 0 || radv_on_driver.load(std::memory_order_acquire) == 0;
    if (!turn) return;
    if (!splash_init()) {
        splash_want.store(0, std::memory_order_release);
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    const int pct = total ? (int)(done * 100 / total) : -1;
    if (splash.last_draw.time_since_epoch().count() != 0 && now - splash.last_draw < std::chrono::milliseconds(250) &&
        pct == splash.last_pct) {
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    if (splash.busy && vkWaitForFences(dev.device, 1, &splash.fence, VK_TRUE, 0) != VK_SUCCESS) {
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    const uint32_t width = dev.swap_extent.width, height = dev.swap_extent.height;
    constexpr uint32_t kBand = 56;
    if (!width || !height || !splash_buffer(width)) {
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    const uint32_t band = std::min(kBand, height);
    const bool bgra = dev.swap_format == VK_FORMAT_B8G8R8A8_UNORM;
    uint8_t bg[4] = {18, 20, 26, 255};
    if (bgra) bg[0] = 26, bg[1] = 20, bg[2] = 18;
    for (uint32_t i = 0; i < width * band; i++) memcpy(splash.map + (size_t)i * 4, bg, 4);
    const int scale = width >= 900 ? 2 : 1;
    splash_text(splash.map, width, band, scale, line, 4, 235, 236, 240);
    if (eta[0]) splash_text(splash.map, width, band, scale, eta, 4 + 8 * scale, 190, 196, 210);
    const int bar_y = 4 + 16 * scale + 4;
    const int bar_h = std::max(6, 4 * scale);
    const int bar_x = (int)width / 10;
    const int bar_w = (int)width - 2 * bar_x;
    auto fill_rect = [&](int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        uint8_t c[4] = {r, g, b, 255};
        if (bgra) c[0] = b, c[1] = g, c[2] = r;
        for (int row = y; row < y + h && row < (int)band; row++) {
            if (row < 0) continue;
            for (int col = x; col < x + w && col < (int)width; col++) {
                if (col < 0) continue;
                memcpy(splash.map + ((size_t)row * width + col) * 4, c, 4);
            }
        }
    };
    if (bar_y + bar_h <= (int)band && bar_w > 0) {
        fill_rect(bar_x, bar_y, bar_w, bar_h, 48, 52, 62);
        if (total) {
            int fw = (int)((uint64_t)bar_w * std::min(done, total) / total);
            if (done && !fw) fw = 1;
            fill_rect(bar_x, bar_y, fw, bar_h, 232, 146, 42);
        }
    }
    if (splash.busy) vkResetFences(dev.device, 1, &splash.fence), splash.busy = false;
    const uint64_t acquire_ns = splash.last_draw.time_since_epoch().count() == 0 ? 100000000ull : 0ull;
    uint32_t index = 0;
    const VkResult got = vkAcquireNextImageKHR(dev.device, dev.swapchain, acquire_ns, splash.acquired, VK_NULL_HANDLE, &index);
    if (got == VK_TIMEOUT || got == VK_NOT_READY || got == VK_ERROR_OUT_OF_DATE_KHR) {
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    const bool acquired = (got == VK_SUCCESS || got == VK_SUBOPTIMAL_KHR) && index < dev.swap_images.size();
    if (!acquired || index >= dev.swap_rendered.size() || !dev.swap_rendered[index]) {
        if (acquired) splash_release(index);
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    vkResetCommandPool(dev.device, splash.pool, 0);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(splash.cmd, &cbbi) != VK_SUCCESS) {
        splash_release(index);
        splash.failed = true;
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    auto barrier = [&](VkImageLayout from, VkImageLayout to) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = dev.swap_images[index];
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(splash.cmd, &di);
    };
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    const VkClearColorValue dark{18 / 255.0f, 20 / 255.0f, 26 / 255.0f, 1.0f};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(splash.cmd, dev.swap_images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dark, 1, &range);
    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy copy{};
    copy.bufferRowLength = width;
    copy.bufferImageHeight = band;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageOffset = {0, (int32_t)((height - band) / 2), 0};
    copy.imageExtent = {width, band, 1};
    vkCmdCopyBufferToImage(splash.cmd, splash.buf, dev.swap_images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    if (vkEndCommandBuffer(splash.cmd) != VK_SUCCESS) {
        splash_release(index);
        splash.failed = true;
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = splash.acquired;
    wait.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = dev.swap_rendered[index];
    signal.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkCommandBufferSubmitInfo cbs{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbs.commandBuffer = splash.cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cbs;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &dev.swap_rendered[index];
    present.swapchainCount = 1;
    present.pSwapchains = &dev.swapchain;
    present.pImageIndices = &index;
    bool submitted = false;
    {
        std::lock_guard qk(dev.present_mu);
        queue_flush_locked();
        submitted = vkQueueSubmit2(dev.queue, 1, &submit, splash.fence) == VK_SUCCESS;
        if (submitted) vkQueuePresentKHR(dev.queue, &present);
    }
    if (!submitted) {
        // A failed submit does not wait on or signal the semaphores. Present the image plainly.
        splash.failed = true;
        splash_release(index);
        splash_yield.store(0, std::memory_order_release);
        return;
    }
    splash.busy = true;
    splash.last_draw = now;
    splash.last_pct = pct;
    splash_want.store(0, std::memory_order_release);
    splash_yield.store(0, std::memory_order_release);
}

void prep_title(size_t done, size_t total, const char* what) {
#ifndef _WIN32
    if (!prep_window || prep_window->version < 2 || !prep_window->set_title || !total) return;
    char title[96];
    snprintf(title, sizeof title, "Hookshot — %s %zu%%", what, done * 100 / total);
    prep_window->set_title(prep_window->window, title);
#else
    (void)done;
    (void)total;
    (void)what;
#endif
}
// (hookshot) hookshot --prepare-shaders (a builder, after a build): the same work with no game, its progress as
// "@@progress" lines on stdout. A skip there leaves hookshot_prepare_skipped beside the game: its own starts then
// leave the work out too (what was not prepared is built when first drawn), until a --prepare-shaders run ends it
bool prepare_only() { return getenv("HOOKSHOT_PREPARE_ONLY") != nullptr; }
std::string skipped_path() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string path(exe, n);
    return path.substr(0, path.find_last_of("\\/") + 1) + "hookshot_prepare_skipped";
}
bool started_up() {
    if (const char* e = getenv("RX_VK_START_DELAY")) Sleep((DWORD)atoi(e) * 1000);   // (diagnosis: a long start, as the first one is)
    if (!replay_start()) return false;
    if (prepare_only()) {
        // (three parts: the programs, the pipelines the notes predict, then those they name; most of the time is the first)
        static int part;
        const auto progress = [](size_t done, size_t total) {
            static size_t last = ~(size_t)0;
            if (!total) return;
            static const size_t from[3] = {0, 700, 900}, span[3] = {700, 200, 100};
            const size_t at = from[part] + done * span[part] / total;
            if (at == last) return;
            last = at;
            printf("@@progress %.3f\n", at / 1000.0);
            fflush(stdout);
        };
        printf("@@stage shaders Preparing shaders\n");
        fflush(stdout);
        part = 0;
        const bool prepared = prepare_run(progress);
        part = 1;
        if (prepared) predict_run(progress);
        part = 2;
        precompile_run(progress);
        pipeline_cache_save(true);
        remove(skipped_path().c_str());
        return true;
    }
    if (FILE* f = fopen(skipped_path().c_str(), "rb")) {
        fclose(f);
        logf("vk: the shaders' preparation was skipped at the build: left to when each is first drawn");
        return true;
    }
    if (precompile_wanted()) {   // (a release build, preparing shaders before the game: nothing to build during play)
        // (the window is already open, but nothing has been presented, so it stays blank until the progress bar below)
        static HANDLE console = nullptr;
        const auto progress = [](size_t done, size_t total) {
            static bool tried = false;
            if (!total) return;
            if (!tried) {
                tried = true;
                if (AllocConsole()) {
#ifdef _WIN32
                    // This console is ours. Selection must not block preparation's progress writes.
                    const HANDLE input = CreateFileA("CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                                     nullptr, OPEN_EXISTING, 0, nullptr);
                    if (input != INVALID_HANDLE_VALUE) {
                        DWORD mode;
                        if (GetConsoleMode(input, &mode))
                            SetConsoleMode(input, (mode & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS);
                        CloseHandle(input);
                    }
#endif
                    SetConsoleTitleA("Hookshot");
                    console = CreateFileA("CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
                    if (console == INVALID_HANDLE_VALUE) console = nullptr;
                }
            }
            splash_progress(done, total);
            if (!console) return;
            char line[80];
            const int n = snprintf(line, sizeof line, "%cPreparing shaders: %zu of %zu (%zu%%)   ", 13, done, total, done * 100 / total);
            DWORD wrote = 0;
            WriteFile(console, line, (DWORD)n, &wrote, nullptr);
        };
        splash_progress(0, 0);
        if (prepare_run(progress)) predict_run(progress);
        precompile_run(progress);
        if (console) CloseHandle(console), console = nullptr, FreeConsole();
        splash_stop();
        prep_title(1, 1, "Hookshot");
#ifdef _WIN32
        if (main_window) SetWindowTextA(main_window, "Hookshot");
#else
        if (prep_window && prep_window->version >= 2 && prep_window->set_title)
            prep_window->set_title(prep_window->window, "Hookshot");
#endif
    }
    jobs_start();
    return true;
}
int start(void* hwnd) {
    main_window = (HWND)hwnd;
    if (!hwnd) {   // (HOS_HEADLESS: no window; the frames are rendered all the same, RX_VK_SHOTS writes them out)
        dev.offscreen = true;
        dev.swap_extent = {1920, 1080};
        dev.swap_format = VK_FORMAT_R8G8B8A8_UNORM;
        logf("vk: no window: presents render offscreen (%ux%u)", dev.swap_extent.width, dev.swap_extent.height);
        return started_up() ? 1 : 0;
    }
#ifdef _WIN32
    VkWin32SurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    si.hinstance = GetModuleHandleA(nullptr);
    si.hwnd = (HWND)hwnd;
    if (vkCreateWin32SurfaceKHR(dev.instance, &si, nullptr, &dev.surface) != VK_SUCCESS) return 0;
    VkBool32 ok = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.queue_family, dev.surface, &ok);
    if (!ok) return 0;
    RECT r{};
    GetClientRect((HWND)hwnd, &r);
    if (!create_swapchain((uint32_t)(r.right - r.left), (uint32_t)(r.bottom - r.top))) return 0;
#else
    const MkwVkWindow* w = (const MkwVkWindow*)hwnd;
    if (w->version != MKWVK_WINDOW_VERSION) {
        logf("vk: unsupported SDL window bridge version %u (expected %u)", w->version, MKWVK_WINDOW_VERSION);
        return 0;
    }
    if (!w->create_surface(w->window, (void*)dev.instance, (void*)&dev.surface)) {
        logf("vk: SDL_Vulkan_CreateSurface failed; check SDL_VIDEODRIVER and the Vulkan window-system extensions");
        return 0;
    }
    VkBool32 ok = VK_FALSE;
    VkResult present = vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.queue_family, dev.surface, &ok);
    if (present != VK_SUCCESS || !ok) {
        logf("vk: queue-family present support failed (VkResult %d, supported %u)", (int)present, ok);
        return 0;
    }
    int width = 0, height = 0;
    prep_window = w;
    w->drawable_size(w->window, &width, &height);
    if (width <= 0 || height <= 0) {
        logf("vk: SDL returned zero drawable size %dx%d", width, height);
        return 0;
    }
    if (!create_swapchain((uint32_t)width, (uint32_t)height)) {
        logf("vk: swapchain creation failed for drawable %dx%d", width, height);
        return 0;
    }
#endif
    if (!started_up()) {
        logf("vk: renderer startup failed after window and swapchain initialization");
        return 0;
    }
    return 1;
}

// (the window's thread: only noted. The swapchain is remade by whoever presents next, in present_image: made here, it went from under a
// frame between its acquire and its submit, and the frame's commands named images of the swapchain just destroyed: a lost device)
void resize(int width, int height) {
    if (dev.offscreen || width <= 0 || height <= 0) return;
    dev.swap_resize.store((uint64_t)(uint32_t)width << 32 | (uint32_t)height, std::memory_order_release);
}

const char* describe() {
    static std::string d;
    d = std::string(dev.props.deviceName) + " (Vulkan, native)";
    return d.c_str();
}

MkwVkRenderer table;

} // namespace
} // namespace mkwvk

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const MkwVkRenderer* mkwvk_init(const MkwVkHost* h) {
    using namespace mkwvk;
    if (!h || h->version != MKWVK_API_VERSION) return nullptr;
    host = h;
    if (!load_global()) {
        logf("vk: vulkan-1.dll not found");
        return nullptr;
    }
    if (!create_instance() || !pick_device() || !create_device()) return nullptr;
    std::set_new_handler([] {
        logf("vk: C++ allocation failed. This is an out-of-memory failure, not \"window closed\" and not a GPU reset.");
        std::abort();
    });
    logf("vk: %s, Vulkan %u.%u, driver %#x; mesh shaders %s, host memory import %s, pipeline libraries %s",
         dev.props.deviceName, VK_VERSION_MAJOR(dev.props.apiVersion), VK_VERSION_MINOR(dev.props.apiVersion),
         dev.props.driverVersion, dev.mesh_shader ? "yes" : "no", dev.external_memory_host ? "yes" : "no",
         dev.graphics_pipeline_library ? "yes" : "no");
    if (radv_driver()) {
        if (pipeline_creates_serialized()) {
            install_radv_driver_thread();
            logf("vk: pipeline creates serialized (RADV), on one driver thread, with shader modules, layouts and the pipeline cache. RX_VK_PIPELINE_SERIAL=0 leaves them parallel");
        }
        const bool vangogh = dev.props.deviceID == 0x163F || std::strstr(dev.props.deviceName, "VANGOGH");
        logf("vk: RADV%s: task/mesh shaders keep the driver optimizer and publish the task payload with output memory",
             vangogh ? " on Van Gogh" : "");
        if (vangogh && !getenv("RX_VK_DECK")) {
#ifndef _WIN32
            setenv("RX_VK_DECK", "1", 0);
            if (!getenv("RX_VK_FRAMEGEN")) setenv("RX_VK_FRAMEGEN", "0", 0);
            if (!getenv("RX_VK_FPS_CAP")) setenv("RX_VK_FPS_CAP", "45", 0);
#endif
            logf("vk: Steam Deck profile (Van Gogh): frame generation off, present cap 45 fps. "
                 "RX_VK_DECK=0 leaves the Insert menu as it is");
        }
    }
    if (low_mem_on(nullptr))
        logf("vk: low-memory mode (Deck or under ~13 GiB). Texture budget and staging are tightened. "
             "RX_VK_LOW_TEX=0, RX_VK_LOW_STAGING=0 and RX_VK_LOW_COMPILE=0 turn those off. RX_VK_LOW_MEM=0 turns them all off.");
    if (!dev.mesh_shader && !dev.mesh_fallback && !getenv("RX_VK_NO_MESH_WARNING")) {
        // (the game draws its foliage, grass and many effects with mesh shaders: without them those draws are left out and the
        // picture is broken (an RX 580's "static"-like scenes, 2026-11-08). Said once, on its own thread, and the game goes on)
        logf("vk: this GPU has no mesh shaders: the game's foliage, grass and many effects cannot be drawn");
#ifdef _WIN32
        std::thread([name = std::string(dev.props.deviceName)] {
            const std::string text = "Your graphics card (" + name + ") or its driver has no mesh shader support, which Ocarina of Time "
                "uses for foliage, grass and many effects, so parts of the picture will be missing or broken.\n\nHookshot needs a GPU "
                "with mesh shaders: NVIDIA GeForce RTX 20 series or newer, AMD Radeon RX 6000 series or newer, or Intel Arc. If your "
                "card is one of those, update its driver.";
            MessageBoxA(nullptr, text.c_str(), "Hookshot: graphics card", MB_OK | MB_ICONWARNING | MB_TOPMOST);
        }).detach();
#endif
    }
    table.version = MKWVK_API_VERSION;
    table.start = start;
    table.resize = resize;
    table.describe = describe;
    fill_replay_table(table);
    return &table;
}

namespace mkwvk {
HWND app_window() { return main_window; }   // (hookshot: the Insert menu's Fullscreen row)

bool radv_driver() {
    // (a call before the device exists must not stick: the id would stay 0 and the lock would never come on)
    if (!dev.physical) return false;
    static int id = 0;
    static bool known = false;
    if (!known) {
        VkPhysicalDeviceDriverProperties dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &dp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        id = (int)dp.driverID;
        known = true;
    }
    return id == VK_DRIVER_ID_MESA_RADV;
}

namespace {
std::mutex pipeline_create_mu;
// Set on threads of ours before they call the driver. A thread that never sets it is the driver's
// own (a compile thread calling back into a hooked entry point) and must not wait on us.
thread_local bool radv_app_thread = false;
void radv_note_app_thread() { radv_app_thread = true; }
}
bool pipeline_creates_serialized() {
    if (!dev.physical) return false;
    static const bool on = [] {
        if (!radv_driver()) return false;
        const char* e = getenv("RX_VK_PIPELINE_SERIAL");
        return !(e && e[0] == '0');
    }();
    return on;
}
PipelineCreateLock::PipelineCreateLock() {
    radv_note_app_thread();
    if (!pipeline_creates_serialized()) return;
    pipeline_create_mu.lock();
    held = true;
}
PipelineCreateLock::~PipelineCreateLock() {
    if (held) pipeline_create_mu.unlock();
}

// (RADV) The prep workers used to take the lock and call the driver on themselves. A compile that keeps going on
// that thread after vkCreate*Pipelines returns is then beside the next worker's call, which is the crash one thread
// never hit. Every shader, layout, pipeline and pipeline-cache call is run here instead, on one thread, so the next
// call joins the last. Translation stays on the workers. The lock above is not taken on this thread.
namespace {
thread_local bool on_radv_thread = false;
std::mutex radv_mu;
std::condition_variable radv_cv;
std::deque<std::function<void()>> radv_q;
std::once_flag radv_once;

void radv_thread_main() {
    on_radv_thread = true;
    SetThreadDescription(GetCurrentThread(), L"mkw_vk radv");
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock lk(radv_mu);
            radv_cv.wait(lk, [] { return !radv_q.empty(); });
            job = std::move(radv_q.front());
            radv_q.pop_front();
        }
        radv_on_driver.fetch_add(1, std::memory_order_acq_rel);
        job();
        radv_on_driver.fetch_sub(1, std::memory_order_acq_rel);
        // Between creates, give the progress bar a moment. Submitting while this thread is inside
        // the ICD waits on the lock that call holds, so the bar used to stay on its first frame.
        if (splash_want.load(std::memory_order_acquire) != 0) {
            static auto last = std::chrono::steady_clock::now() - std::chrono::seconds(3);
            const auto now = std::chrono::steady_clock::now();
            if (now - last >= std::chrono::seconds(2)) {
                last = now;
                splash_yield.store(1, std::memory_order_release);
                const auto until = now + std::chrono::milliseconds(100);
                while (splash_yield.load(std::memory_order_acquire) != 0 && std::chrono::steady_clock::now() < until)
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                splash_yield.store(0, std::memory_order_release);
            }
        }
    }
}
void radv_ensure_thread() {
    std::call_once(radv_once, [] { std::thread(radv_thread_main).detach(); });
}
template <class F>
auto radv_drive(F fn) -> decltype(fn()) {
    using R = decltype(fn());
    // Re-entry from this thread runs here. Every other call, including one from a thread that never
    // took PipelineCreateLock, waits its turn on this thread. The hooks are only our vkx:: table, so
    // a RADV compile thread does not call them; running an unmarked thread of ours inline was the
    // concurrency the driver thread exists to stop. The log says when that happens.
    if (on_radv_thread) return fn();
    if (!radv_app_thread) {
        static std::atomic<int> told{0};
        if (told.fetch_add(1, std::memory_order_relaxed) < 4)
            logf("vk: radv: an unmarked thread's driver call is queued on the driver thread");
    }
    radv_ensure_thread();
    auto prom = std::make_shared<std::promise<R>>();
    std::future<R> fut = prom->get_future();
    {
        std::lock_guard lk(radv_mu);
        radv_q.push_back([fn, prom]() mutable {
            try {
                if constexpr (std::is_void_v<R>) {
                    fn();
                    prom->set_value();
                } else {
                    prom->set_value(fn());
                }
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
        });
    }
    radv_cv.notify_one();
    return fut.get();
}

#define RADV_RESULT(name, formal, actual)                                                                                  \
    PFN_##name real_##name = nullptr;                                                                                      \
    VkResult VKAPI_CALL name##_thr formal { return radv_drive([=] { return real_##name actual; }); }
#define RADV_VOID(name, formal, actual)                                                                                    \
    PFN_##name real_##name = nullptr;                                                                                      \
    void VKAPI_CALL name##_thr formal { radv_drive([=] { real_##name actual; }); }

RADV_RESULT(vkCreateShaderModule,
            (VkDevice device, const VkShaderModuleCreateInfo* info, const VkAllocationCallbacks* alloc, VkShaderModule* out),
            (device, info, alloc, out))
RADV_VOID(vkDestroyShaderModule, (VkDevice device, VkShaderModule mod, const VkAllocationCallbacks* alloc), (device, mod, alloc))
RADV_RESULT(vkCreateDescriptorSetLayout,
            (VkDevice device, const VkDescriptorSetLayoutCreateInfo* info, const VkAllocationCallbacks* alloc, VkDescriptorSetLayout* out),
            (device, info, alloc, out))
RADV_VOID(vkDestroyDescriptorSetLayout, (VkDevice device, VkDescriptorSetLayout layout, const VkAllocationCallbacks* alloc),
          (device, layout, alloc))
RADV_RESULT(vkCreatePipelineLayout,
            (VkDevice device, const VkPipelineLayoutCreateInfo* info, const VkAllocationCallbacks* alloc, VkPipelineLayout* out),
            (device, info, alloc, out))
RADV_VOID(vkDestroyPipelineLayout, (VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks* alloc), (device, layout, alloc))
RADV_RESULT(vkCreateGraphicsPipelines,
            (VkDevice device, VkPipelineCache cache, uint32_t n, const VkGraphicsPipelineCreateInfo* info, const VkAllocationCallbacks* alloc,
             VkPipeline* out),
            (device, cache, n, info, alloc, out))
RADV_RESULT(vkCreateComputePipelines,
            (VkDevice device, VkPipelineCache cache, uint32_t n, const VkComputePipelineCreateInfo* info, const VkAllocationCallbacks* alloc,
             VkPipeline* out),
            (device, cache, n, info, alloc, out))
RADV_VOID(vkDestroyPipeline, (VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* alloc), (device, pipeline, alloc))
RADV_RESULT(vkCreatePipelineCache,
            (VkDevice device, const VkPipelineCacheCreateInfo* info, const VkAllocationCallbacks* alloc, VkPipelineCache* out),
            (device, info, alloc, out))
RADV_VOID(vkDestroyPipelineCache, (VkDevice device, VkPipelineCache cache, const VkAllocationCallbacks* alloc), (device, cache, alloc))
RADV_RESULT(vkGetPipelineCacheData, (VkDevice device, VkPipelineCache cache, size_t* size, void* data), (device, cache, size, data))
RADV_RESULT(vkCreateSampler, (VkDevice device, const VkSamplerCreateInfo* info, const VkAllocationCallbacks* alloc, VkSampler* out),
            (device, info, alloc, out))
RADV_VOID(vkDestroySampler, (VkDevice device, VkSampler sampler, const VkAllocationCallbacks* alloc), (device, sampler, alloc))
#undef RADV_RESULT
#undef RADV_VOID
} // namespace

void install_radv_driver_thread() {
    static bool installed = false;
    if (installed || !pipeline_creates_serialized()) return;
    installed = true;
    radv_note_app_thread(); // the renderer thread (prep, the precompile's load, presents)
#define RADV_INSTALL(name)                                                                                                 \
    do {                                                                                                                   \
        if (vkx::name) {                                                                                                   \
            real_##name = vkx::name;                                                                                       \
            vkx::name = &name##_thr;                                                                                       \
        }                                                                                                                  \
    } while (0)
    RADV_INSTALL(vkCreateShaderModule);
    RADV_INSTALL(vkDestroyShaderModule);
    RADV_INSTALL(vkCreateDescriptorSetLayout);
    RADV_INSTALL(vkDestroyDescriptorSetLayout);
    RADV_INSTALL(vkCreatePipelineLayout);
    RADV_INSTALL(vkDestroyPipelineLayout);
    RADV_INSTALL(vkCreateGraphicsPipelines);
    RADV_INSTALL(vkCreateComputePipelines);
    RADV_INSTALL(vkDestroyPipeline);
    RADV_INSTALL(vkCreatePipelineCache);
    RADV_INSTALL(vkDestroyPipelineCache);
    RADV_INSTALL(vkGetPipelineCacheData);
    RADV_INSTALL(vkCreateSampler);
    RADV_INSTALL(vkDestroySampler);
#undef RADV_INSTALL
    radv_ensure_thread();
}

bool radv_mesh_payload_barrier() {
    const char* e = getenv("RX_VK_MESH_PAYLOAD_BARRIER");
    if (e && e[0]) return std::atoi(e) != 0;
    return radv_driver();
}

bool low_mem_on(const char* off_switch) {
    static const int mode = [] {
        const char* e = getenv("RX_VK_LOW_MEM");
        if (e && e[0]) return std::atoi(e) != 0 ? 1 : 0;
        if (const char* d = getenv("RX_VK_DECK"); d && d[0] != '0') return 1;
#ifndef _WIN32
        if (FILE* f = fopen("/proc/meminfo", "r")) {
            char line[128];
            unsigned long long kb = 0;
            while (fgets(line, sizeof line, f))
                if (sscanf(line, "MemTotal: %llu", &kb) == 1) break;
            fclose(f);
            if (kb && kb < 13ull * 1024 * 1024) return 1;
        }
#endif
        return 0;
    }();
    if (!mode) return false;
    const char* e = off_switch ? getenv(off_switch) : nullptr;
    return !(e && e[0] == '0');
}
} // namespace mkwvk
