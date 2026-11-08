// Vulkan Memory Allocator (v3.3.0, from Eglantine's CPM cache: nx2/build.py adds its include path): images and
// device buffers are sub-allocated from large blocks (NVIDIA allows ~4096 allocations, and the imported pools
// already take about 2400 of them). Its Vulkan functions come from vkGetInstanceProcAddr / vkGetDeviceProcAddr.
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vk_internal.h"
#include <vk_mem_alloc.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace mkwvk {

VmaAllocator vma;

namespace {
constexpr int kTags = (int)MemTag::Count;
std::atomic<uint64_t> tag_bytes[kTags]{};
std::atomic<uint64_t> tag_count[kTags]{};
int tag_index(MemTag tag) {
    const int t = (int)tag;
    return t >= 0 && t < kTags ? t : (int)MemTag::Other;
}
void mem_tag_add(VmaAllocation allocation, MemTag tag) {
    if (!allocation) return;
    const int t = tag_index(tag);
    vmaSetAllocationUserData(vma, allocation, (void*)(uintptr_t)(t + 1));
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(vma, allocation, &info);
    tag_bytes[t].fetch_add(info.size, std::memory_order_relaxed);
    tag_count[t].fetch_add(1, std::memory_order_relaxed);
}
void mem_tag_sub(VmaAllocation allocation) {
    if (!allocation) return;
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(vma, allocation, &info);
    int t = info.pUserData ? (int)(uintptr_t)info.pUserData - 1 : (int)MemTag::Other;
    if (t < 0 || t >= kTags) t = (int)MemTag::Other;
    tag_bytes[t].fetch_sub(info.size, std::memory_order_relaxed);
    tag_count[t].fetch_sub(1, std::memory_order_relaxed);
}
} // namespace

void mem_tag_move(VmaAllocation allocation, MemTag tag) {
    if (!allocation) return;
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(vma, allocation, &info);
    const int next = tag_index(tag);
    const int cur = info.pUserData ? (int)(uintptr_t)info.pUserData - 1 : (int)MemTag::Other;
    if (cur == next) return;
    if (cur >= 0 && cur < kTags) {
        tag_bytes[cur].fetch_sub(info.size, std::memory_order_relaxed);
        tag_count[cur].fetch_sub(1, std::memory_order_relaxed);
    }
    vmaSetAllocationUserData(vma, allocation, (void*)(uintptr_t)(next + 1));
    tag_bytes[next].fetch_add(info.size, std::memory_order_relaxed);
    tag_count[next].fetch_add(1, std::memory_order_relaxed);
}
uint64_t mem_tag_bytes(MemTag tag) { return tag_bytes[tag_index(tag)].load(std::memory_order_relaxed); }
uint64_t mem_tag_count(MemTag tag) { return tag_count[tag_index(tag)].load(std::memory_order_relaxed); }
void mem_tag_format(char* buf, size_t n) {
    static const char* name[kTags] = {"textures", "targets", "captures", "staging", "streams", "bc scratch",
                                      "page table", "download", "present", "debug", "other"};
    int used = snprintf(buf, n, "vk: allocations by kind:");
    for (int i = 0; i < kTags && used > 0 && (size_t)used < n; i++)
        used += snprintf(buf + used, n - (size_t)used, " %s %llu MB (%llu)", name[i],
                         (unsigned long long)(tag_bytes[i].load(std::memory_order_relaxed) >> 20),
                         (unsigned long long)tag_count[i].load(std::memory_order_relaxed));
}

bool vma_init(PFN_vkGetInstanceProcAddr gipa) {
    VmaVulkanFunctions f{};
    f.vkGetInstanceProcAddr = gipa;
    f.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo ci{};
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
               (dev.memory_budget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0);
    ci.physicalDevice = dev.physical;
    ci.device = dev.device;
    ci.instance = dev.instance;
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    ci.pVulkanFunctions = &f;
    ci.preferredLargeHeapBlockSize = 64ull << 20;   // (64 MB blocks: Eglantine found 256 MB ones wasted VRAM)
    return vmaCreateAllocator(&ci, &vma) == VK_SUCCESS;
}

// Images come from a pool of 128 MB blocks: on their own, VMA gave most of them dedicated memory (the driver prefers it
// for render targets, and anything over half a default block), a vkAllocateMemory and a kernel call for each, and
// the dynamic resolution makes a new set of render targets every few frames. Images too large for a block, or of a
// memory type the pool's is not, are allocated as before.
VmaPool image_pool = VK_NULL_HANDLE, rt_pool = VK_NULL_HANDLE;
bool image_pool_tried = false, rt_pool_tried = false;
// (RX_VK_IMAGE_BLOCK_MB=<n>, 16 and up. RADV in low-memory mode uses 32 MB: amdgpu does not have NVIDIA's ~4096
// allocation cap, and 128 MB blocks kept a gigabyte free inside them. Everyone else stays at 128 MB)
uint64_t image_block_bytes() {
    static uint64_t v = 0;
    if (!v) {
        if (const char* e = getenv("RX_VK_IMAGE_BLOCK_MB")) v = (uint64_t)std::max(16, atoi(e)) << 20;
        else if (radv_driver() && low_mem_on(nullptr)) v = 32ull << 20;
        else v = 128ull << 20;
    }
    return v;
}
// (RX_VK_DEDICATED_MB=<n>: images at least this big get their own memory object. 0: never. Unset, on RADV in
// low-memory mode: half a block, so freeing one gives the memory back whole)
uint64_t dedicated_floor() {
    if (const char* e = getenv("RX_VK_DEDICATED_MB")) {
        if (!e[0] || (e[0] == '0' && e[1] == 0)) return ~0ull;
        return (uint64_t)std::max(1, atoi(e)) << 20;
    }
    if (!(radv_driver() && low_mem_on(nullptr))) return ~0ull;
    return image_block_bytes() / 2;
}
bool rt_pool_on() {
    if (const char* e = getenv("RX_VK_RT_POOL")) return e[0] != '0';
    return radv_driver() && low_mem_on(nullptr);
}
uint64_t image_requirement_bytes(const VkImageCreateInfo& ci) {
    if (!vkGetDeviceImageMemoryRequirements) return 0;
    VkDeviceImageMemoryRequirements req{VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS};
    req.pCreateInfo = &ci;
    VkMemoryRequirements2 mr{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(dev.device, &req, &mr);
    return mr.memoryRequirements.size;
}
VmaPool make_pool(VmaPool* slot, bool* tried, const char* what) {
    if (*tried) return *slot;
    *tried = true;
    VkImageCreateInfo rep{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    rep.imageType = VK_IMAGE_TYPE_2D;
    rep.format = VK_FORMAT_R8G8B8A8_UNORM;
    rep.extent = {256, 256, 1};
    rep.mipLevels = rep.arrayLayers = 1;
    rep.samples = VK_SAMPLE_COUNT_1_BIT;
    rep.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type = 0;
    if (vmaFindMemoryTypeIndexForImageInfo(vma, &rep, &ai, &type) != VK_SUCCESS) return VK_NULL_HANDLE;
    VmaPoolCreateInfo pi{};
    pi.memoryTypeIndex = type;
    pi.blockSize = image_block_bytes();
    if (vmaCreatePool(vma, &pi, slot) != VK_SUCCESS) *slot = VK_NULL_HANDLE;
    logf("vk: %s from a pool of %llu MB blocks (memory type %u)%s", what, (unsigned long long)(image_block_bytes() >> 20), type,
         *slot ? "" : ": not made");
    return *slot;
}
VmaPool images_pool() { return make_pool(&image_pool, &image_pool_tried, "images"); }
VmaPool render_target_pool() { return make_pool(&rt_pool, &rt_pool_tried, "render targets"); }

bool create_image(const VkImageCreateInfo& ci, VkImage& image, VmaAllocation& allocation, bool dedicated, MemTag tag) {
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    // (in VRAM or not at all: an image VMA puts in system memory when VRAM is full is read over PCIe every frame,
    // and takes the RAM the game and everything else on the machine need)
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (dev.memory_budget) ai.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    if (!dedicated) {
        const uint64_t floor = dedicated_floor();
        if (floor != ~0ull) {
            const uint64_t need = image_requirement_bytes(ci);
            if (need >= floor) dedicated = true;
        }
    }
    if (dedicated) {   // (a memory object of its own, which freeing gives back whole: a pool's block is kept by VMA when it empties)
        VmaAllocationCreateInfo dai = ai;
        dai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        if (vmaCreateImage(vma, &ci, &dai, &image, &allocation, nullptr) == VK_SUCCESS) return mem_tag_add(allocation, tag), true;
    }
    static const bool no_pool = getenv("RX_VK_NO_IMAGE_POOL") != nullptr;
    const bool use_rt = tag == MemTag::RenderTarget && rt_pool_on();
    if (VmaPool pool = no_pool ? VK_NULL_HANDLE : (use_rt ? render_target_pool() : images_pool())) {
        VmaAllocationCreateInfo pai = ai;
        pai.pool = pool;
        pai.usage = VMA_MEMORY_USAGE_UNKNOWN;
        if (vmaCreateImage(vma, &ci, &pai, &image, &allocation, nullptr) == VK_SUCCESS) return mem_tag_add(allocation, tag), true;
    }
    if (vmaCreateImage(vma, &ci, &ai, &image, &allocation, nullptr) == VK_SUCCESS) return mem_tag_add(allocation, tag), true;
    // (hookshot) VRAM is full (a 4 GB card at a high resolution, #96): an image left unmade is a black screen, so it goes
    // to system memory instead, slow but drawn
    if (getenv("RX_VK_IMAGES_VRAM_ONLY")) return false;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(vma, &ci, &sai, &image, &allocation, nullptr) != VK_SUCCESS) return false;
    mem_tag_add(allocation, tag);
    static std::atomic<int> told{0};
    if (told++ < 4) logf("vk: VRAM full: a %ux%u image (format %d) in system memory", ci.extent.width, ci.extent.height, (int)ci.format);
    return true;
}
void destroy_image(VkImage image, VmaAllocation allocation) {
    mem_tag_sub(allocation);
    vmaDestroyImage(vma, image, allocation);
}
uint64_t allocation_size(VmaAllocation allocation) {
    VmaAllocationInfo info{};
    if (allocation) vmaGetAllocationInfo(vma, allocation, &info);
    return info.size;
}
bool create_buffer(uint64_t size, VkBufferUsageFlags usage, bool host_visible, VkBuffer& buffer,
                   VmaAllocation& allocation, void** mapped, bool dedicated, MemTag tag) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    if (host_visible) ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    static const bool blocks = getenv("RX_VK_BLOCK_BUFFERS") != nullptr;
    if (dedicated && !blocks) ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vma, &bi, &ai, &buffer, &allocation, &info) != VK_SUCCESS) return false;
    if (mapped) *mapped = info.pMappedData;
    mem_tag_add(allocation, tag);
    return true;
}
/// A buffer in cached host memory, mapped: written on one thread, read back on the CPU too (the captures: the game's
/// threads copy pages into it, the replay reads texture pages out of it and the GPU copies the rest from it)
bool create_buffer_cached(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation,
                          void** mapped, bool dedicated, MemTag tag) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (dedicated) ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;   // (its own memory object: no part of a 64 MB block left unused)
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vma, &bi, &ai, &buffer, &allocation, &info) != VK_SUCCESS) return false;
    if (mapped) *mapped = info.pMappedData;
    mem_tag_add(allocation, tag);
    return true;
}
void flush_mapped(VmaAllocation allocation, uint64_t size) {
    if (size) vmaFlushAllocation(vma, allocation, 0, size);
}
void invalidate_mapped(VmaAllocation allocation, uint64_t size) {
    if (size) vmaInvalidateAllocation(vma, allocation, 0, size);
}
void destroy_buffer(VkBuffer buffer, VmaAllocation allocation) {
    mem_tag_sub(allocation);
    vmaDestroyBuffer(vma, buffer, allocation);
}
bool create_device_buffer(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, bool dedicated,
                          MemTag tag) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (dev.memory_budget) ai.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    if (dedicated) ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    if (vmaCreateBuffer(vma, &bi, &ai, &buffer, &allocation, nullptr) != VK_SUCCESS) return false;
    mem_tag_add(allocation, tag);
    return true;
}
/// (every heap: what VMA has in blocks and allocations there, what the process uses of it, its budget; one line)
void vma_log_heaps() {
    VmaBudget b[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma, b);
    char line[1024];
    int n = snprintf(line, sizeof line, "vk: memory heaps:");
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount && n < 900; h++)
        n += snprintf(line + n, sizeof line - n, " heap %u%s: %u blocks %llu MB, %u allocations %llu MB; used %llu of %llu MB;", h,
                      (dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " (device)" : " (system)",
                      b[h].statistics.blockCount, (unsigned long long)(b[h].statistics.blockBytes >> 20),
                      b[h].statistics.allocationCount, (unsigned long long)(b[h].statistics.allocationBytes >> 20),
                      (unsigned long long)(b[h].usage >> 20), (unsigned long long)(b[h].budget >> 20));
    logf("%s", line);
}
bool vram_usage(uint64_t* used, uint64_t* budget, uint64_t* block_free) {
    VmaBudget b[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma, b);
    uint64_t u = 0, t = 0, f = 0;
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount; h++)
        if (dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            u += b[h].usage, t += b[h].budget;
            // (the free space in VMA's blocks: the driver counts it as the process's, new allocations take it first)
            f += b[h].statistics.blockBytes - std::min(b[h].statistics.blockBytes, b[h].statistics.allocationBytes);
        }
    *used = u, *budget = t;
    if (block_free) *block_free = f;
    return t != 0;
}

void vma_heap_account(uint64_t* dev_used, uint64_t* dev_blocks, uint64_t* dev_alloc, uint64_t* dev_free,
                      uint64_t* sys_used, uint64_t* sys_blocks, uint64_t* sys_alloc) {
    VmaBudget b[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma, b);
    uint64_t du = 0, db = 0, da = 0, su = 0, sb = 0, sa = 0;
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount; h++) {
        const auto& s = b[h].statistics;
        if (dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            du += b[h].usage, db += s.blockBytes, da += s.allocationBytes;
        else
            su += b[h].usage, sb += s.blockBytes, sa += s.allocationBytes;
    }
    if (dev_used) *dev_used = du;
    if (dev_blocks) *dev_blocks = db;
    if (dev_alloc) *dev_alloc = da;
    if (dev_free) *dev_free = db - std::min(db, da);
    if (sys_used) *sys_used = su;
    if (sys_blocks) *sys_blocks = sb;
    if (sys_alloc) *sys_alloc = sa;
}

/// The system-memory heaps the GPU reaches: what the process uses there (the imported guest memory too: pinned by the
/// driver) and what the driver allows it, asked now (a new frame index makes VMA ask the driver again); false when unknown
bool system_heap_usage(uint64_t* used, uint64_t* budget) {
    static uint32_t frame = 0;
    vmaSetCurrentFrameIndex(vma, ++frame);
    VmaBudget b[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma, b);
    uint64_t u = 0, t = 0;
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount; h++)
        if (!(dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) u += b[h].usage, t += b[h].budget;
    *used = u, *budget = t;
    return dev.memory_budget && t != 0;
}

} // namespace mkwvk
