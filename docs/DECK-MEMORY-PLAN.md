# Steam Deck memory plan

Planning only, from `cursor/deck-radv-mesh-ed6e` at `5717fb0`. No game code changes in this document. Another agent is fixing bugs on that branch; this plan lives on `cursor/deck-memory-plan` and should not be pushed there.

The machine is a Steam Deck OLED: 11.7 GB of RAM shared by the CPU and the GPU (RADV Van Gogh, `deviceID` 0x163F), about 6.7 GB of swap, a 4-core/8-thread Zen 2, SteamOS 3.9, earlyoom. Hookshot is a native port of the Switch 2 *Ocarina of Time* remake. It lifts the game's AArch64 to C (`-O1`, `-DA64_POLLING`, `-DA64_WRITE_TRACK`), replays NVN2 on Vulkan (`gpu/vk/vk_*.cpp`, task/mesh shaders), and reads an extracted romfs (about 26 GB, 186k files).

The Deck profile already in `5717fb0` turns on when `RX_VK_DECK` is set or `/proc/meminfo` `MemTotal` is under about 13 GiB, unless `RX_VK_LOW_MEM=0`:

| Knob | Deck value | Where |
|---|---|---|
| Texture budget | 1280 MB, top mips dropped from 1024 px / 256 MB / 32 MB | `vk_texture.cpp` `RX_VK_LOW_TEX` |
| Captures | 6 kept, 96 MB | `vk_memory.cpp` `RX_VK_LOW_CAPTURES` |
| Program cache | private mmap, `POSIX_MADV_DONTNEED` on payloads | `vk_shader.cpp` `RX_VK_LOW_PROGRAMS` |
| Pipeline cache | mapped only for `vkCreatePipelineCache`, then unmapped | `vk_draw.cpp` `RX_VK_LOW_PIPELINES` |
| Staging kept | 48 MB | `vk_texture.cpp` `RX_VK_LOW_STAGING` |
| Import chunk | 8 MB | `vk_memory.cpp` `RX_VK_LOW_IMPORT` |
| Mirror budget | 256 MB, and mirrors stay off on an iGPU unless `RX_VK_MIRROR=1` | `vk_memory.cpp` `RX_VK_LOW_MIRROR` |
| Resolution | Auto becomes 50% | `vk_menu.cpp` |
| Present cap | 45 fps, frame generation off | `vk_main.cpp` |

That pass is necessary and not sufficient. The run below still died on the first outdoor load.

## What the Deck run actually measured

Build `5717fb0`, first run, fresh caches:

- RSS 3.3 GB at the title, a 6.8 GB peak about two minutes in, then 4–5 GB.
- Swap free fell from 5.3 GB to 0 and did not come back when RSS fell.
- After the process was killed, about 5.4 GB of RAM and 2.8 GB of swap came back. The live footprint was about 8 GB, not the 4.2 GB RSS earlyoom printed.
- 1–2 fps while walking, CPU about 75% idle, GPU mostly idle, swap in and out about 2 MB/s each. That is the working set larger than RAM, not a shader or a CPU limit.
- earlyoom SIGTERMed it at the moment Link left the house (`mem 3.40%, swap 5.79%, VmRSS 4188 MiB`). That is the first Kokiri Forest load.
- `hookshot_programs.bin` (5.8 GB) was rewritten whole twice. `hookshot_shaders.bin` is 512 MB, `mkw_vk_pipelines.bin` grew to 77 MB, `hookshot_pipelines.bin` to 27 MB.
- Two threads named like `mkw_vk` sat at about 95% CPU early. Those are the background workers in `vk_jobs.cpp` (`hardware_concurrency()/3`, clamped to 2–4; on this CPU that is 2).
- VSZ was 526 GB.
- Link drew as a glowing white silhouette. The room did not. See the last section.

VSZ is not a leak. `reserve_guest` in `host/hos_main.c` maps `HOS_GUEST_SPACE` (1 << 39, 512 GB) at `A64_HOST_BASE` (`0x100000000000`) with `MAP_NORESERVE`. The other ~14 GB is the executable, thread stacks, the program-cache map, and ordinary heap. earlyoom does not look at VSZ. Do not spend a change on shrinking the reservation.

RSS falling from 6.8 GB to 4–5 GB while swap free went to zero means those pages are still allocated. Linux does not bring a swapped page back until it is touched, and it does not drop a swapped anonymous page until it is freed. The 1–2 fps walk is the game touching that set faster than the disk can move it (2 MB/s in and 2 MB/s out on a Zen 2 with the SSD also taking the shader-cache writes).

SteamOS earlyoom kills when available RAM and free swap are both under its thresholds. The line above is already under the usual 10%/10% pair (`MemAvailable` about 400 MB, swap free about 400 MB). A budget that is only an RSS cap will still die, because swapped and pinned pages count. The number to stay under is resident + swapped + pinned, and `MemAvailable` has to stay above roughly 1.5 GB so a 10% earlyoom line is not the thing that stops the game.

On Van Gogh there is one memory pool. A `DEVICE_LOCAL` image, a pinned userptr import, and an anonymous guest page are the same RAM. `VK_EXT_memory_budget` on this APU reports a budget on the order of the whole heap, not a 4 GB discrete card. The existing "under 6 GB allowed, so BC-compress" test in `vk_texture.cpp` therefore does not trip here.

## 1. Memory map

Sizes are from the code plus the one Deck run. Anything not printed by that run is marked *estimate* and has a measurement that replaces it. "Resident" means pages in RAM. "Pinned" means the kernel will not swap them (amdgpu userptr).

### Guest window

`host/hos_main.c` `reserve_guest`, `host/hos.h`.

| Region | Guest address | What commits it | Size |
|---|---|---|---|
| System objects | `0x01000000`–`0x07000000` | `hos_mem_init` calls `hos_commit` on all 96 MB. On Linux that commit is a no-op (below). Pages become resident when the main stack and TLS are touched. | 96 MB virtual, *estimate* under 16 MB resident |
| Image | `hos_image_base` (`0x80000000` in Eden's layout) for `image.bin` | The boot reads the whole file into the window (`hos_main.c`, the comment about not reading "the rest of the window"). Every byte is touched. | The file size. Logged as `boot: image %zu MB`. The lifted executable is a separate 426 MB (`docs/design.md`); this file is the guest image. |
| Heap | `0x1000000000` upward | `SetMemoryHeapSize` then the game's writes. The game is told `HOS_MEMORY_MB` or **9809 MB** (`hos_mem.c`, the Switch 2 figure). | Whatever the game has touched and not freed. The forest import note below says Kokiri's GPU pools alone are about 1.3 GB of this. |
| Address regions | Down from `0x8000000000` | `AllocateAddressRegion` / `AllocateMemoryPages`. Freed with `MADV_DONTNEED`. | Scene streaming. Same physical pool as the heap. |

`hos_commit` on Linux (`host/posix/win32_posix.c` `VirtualAlloc`) returns immediately when the address is inside the guest window. `MAP_NORESERVE` already made the pages legal. Physical memory appears on the first store, not on the commit call. `HOS_MEM_MAP=1` is wrong on Linux: `VirtualQuery` reports the entire 512 GB as one `MEM_COMMIT` region, so that log line will say 524288 MB committed. Do not use it.

`FreeMemoryPages` and `hos_region_free` do call `MADV_DONTNEED` (`hos_mem.c`). Freed guest pages can leave RAM. Two gaps:

- The game is told it owns 9809 MB on an 11.7 GB machine. A Switch 2 title sizes its streaming heap from `QueryMemoryInfo` (total at `+0x00`, used at `+0x08` = image + 16 MB). If OoT does what the TOTK comment in that function describes, it will try to own several gigabytes before the first outdoor scene.
- Nothing accounts touched-but-live pages. A heap the game never frees stays resident or swapped for the rest of the process.

`A64_WRITE_TRACK` (`third_party/a64-kit/rt/a64.h`) is a 1 MB bitmap, `a64_dirty[1 << 20]`, indexed by `(uint32_t)addr >> 12`. That is 4 GB of addresses, not 512 GB. It costs an atomic store on every non-stack guest store. It is not a multi-gigabyte allocation. The page-generation tables in `vk_memory.cpp` (`kPages = 1 << 27`) are two 1 MB region arrays plus 256 KB chunks created when a 256 MB span is first stamped. A fully written 4 GB heap is about 4 MB of tables.

**Measure:** one line of `/proc/self/smaps` for the VMA at `0x100000000000`. `Rss`, `Pss`, `Swap`, `Locked`, `Size`. Split is not available inside that one VMA; add a counter beside `SetMemoryHeapSize` / `AllocateMemoryPages` / `FreeMemoryPages` (bytes touched, bytes `MADV_DONTNEED`'d). The boot log already prints image size and each `heap: N MB`.

### Lifted code

`docs/design.md`: 294,450 functions, 1.4 GB of C, `hookshot.exe` about 426 MB, compiled `-O1`. The text is file-backed. Cold functions are reclaimable. A play session that only runs the opening and Kokiri will not make all 426 MB hot; *estimate* 150–300 MB resident once the scene is warm, more while new code is faulted in during the load.

**Measure:** smaps of the executable mapping (`r-x`). `Rss` versus `Size`.

### Romfs

`host/nn_fs.c` reads with positional `ReadFile`. There is no userspace file cache. The copy that matters is the one the game makes in the guest heap. The kernel keeps a second copy in the page cache, plus dentries and inodes for a 186k-file tree.

Page cache is reclaimable and is inside `MemAvailable`, so earlyoom will not count clean cache as pressure by itself. It still loses the race when a scene load faults guest pages and reads thousands of files in the same second: reclaim is not instantaneous, and dirty cache (the program-cache writes) is not reclaimable at all. *Estimate* for a scene load: a few hundred MB of clean cache and, if the game stats a large directory, tens to a couple of hundred MB of dentry/inode slab. Not the 8 GB, but it is part of why `MemAvailable` hits 3% while RSS is only 4.2 GB.

**Measure:** `/proc/meminfo` `Cached`, `Dirty`, `SReclaimable`, and `slabtop` or `/proc/slabinfo` for `dentry`/`inode` before and after the door. Not in the process RSS.

### Vulkan, VMA, and the UMA heap

`gpu/vk/vk_vma.cpp`. Images come from a pool of 128 MB `DEVICE_LOCAL` blocks. Other device buffers use VMA's 64 MB blocks. `AUTO_PREFER_DEVICE` asks for `DEVICE_LOCAL`, which on Van Gogh is system RAM. If that allocation fails, images fall back to system memory anyway (`RX_VK_IMAGES_VRAM_ONLY` is off). There is no second copy avoided by that fallback; both copies are the same RAM, and the guest original is still there.

Host imports (`vk_memory.cpp` `import_pool`, `VK_EXT_external_memory_host`) do **not** copy. They bind a VkBuffer at the guest pointer. On RADV that binding is an amdgpu userptr, and the comment at the import reaper is the important one: the kernel pins those pages and walks every page of every live userptr at each submit. Retired imports are freed now (`dead_imports`), which fixed a submit-time walk that had grown without bound. The pages of a **live** import cannot be swapped. Kokiri Forest is called out in that comment as **about 1.3 GB of imports**.

Mirrors (`RX_VK_MIRROR`) duplicate a pool into `DEVICE_LOCAL` for discrete GPUs. `mirror_on()` is false unless the device is discrete or `RX_VK_MIRROR=1`. On the Deck the low-mem 256 MB cap is a backstop, not the steady state. Leave it off. Turning it on would add a second copy of the 1.3 GB.

**Measure:** `vma_log_heaps` (already every 600 presents; too rare, see instrumentation). Sum of live import sizes (log it). smaps `Locked` on the guest VMA, and `/proc/meminfo` `Unevictable`.

### Textures, render targets, staging

`vk_texture.cpp`.

- Budget: 70% of the driver's reported budget, clamped to 6 GB, then **clamped again to 1280 MB** in low-mem mode. Eviction runs every 10 frames, only down to 7/8 of the budget, and will not touch anything used in the last 30 frames (`RX_VK_EVICT_MIN_UNUSED`).
- The 1280 MB is not a hard ceiling during a load. Eviction is skipped while `total_bytes` is under the budget, and a scene change uploads the new set before the old set has been idle for 30 frames. Both sets are live together.
- ASTC is decoded to RGBA8 (4 bytes/texel) unless it qualifies for BC3. Large ASTC (6x6 and up, or anything over 2 MB) can become BC3. Finer textures, which the comment names as UI and characters, stay RGBA8 until they are over 64 MB, unless the driver budget is under 6 GB. On this APU that second condition is false, so character-sized textures stay expanded about 4x versus BC3 and about 8x versus the ASTC in guest memory.
- Top mip levels are dropped for ASTC once a texture would pass 1024 px and 32 MB, and for any texture once it would pass 256 MB. Render targets are not ASTC. A dynamic-resolution set that the GPU has drawn is kept (the comment records 3.6 GB of stale targets before the overwrite eviction). Low-mem shortens that keep from 3600 presents to 900, and an overwritten target can go. A target that is merely unused is kept. Several resolutions at once still fit inside 1280 MB and delay eviction of real textures.
- Staging arenas are 64 MB each, allocated to the size of the upload, freed at frame end past 48 MB kept (and any arena over 64 MB). The comment above that trim says a scene change stages hundreds of MB to over a GB **before** the trim runs. That spike is inside the load that killed the process, and it is on top of the textures.

**Measure:** the existing line `textures %llu MB`, plus the ASTC / rendered / other split already computed every 600 frames. Add peak `total_bytes` and peak staging since the last log, not the post-trim value.

### Captures

`vk_memory.cpp`. Each submit copies guest pages written since the last submit into a host-visible buffer (minimum allocation 16 MB, `cap_buf_get`). Low-mem keeps 6 of them and 96 MB. Spare empty buffers: 2. This is a real copy, on top of the guest pages, and it is `DEVICE_LOCAL` or host-visible VMA memory, so it is pinned for the GPU. 96 MB is small next to 8 GB. Do not start here. The capture table (`CapEntry` chunks, 512 KB per 256 MB of guest) is noise.

**Measure:** `cap_bytes` (add it to the memory line; it is not there today).

### Shader, program, and pipeline caches

| File | Size on the run | Resident? |
|---|---|---|
| `hookshot_shaders.bin` | 512 MB | Whole file is `fread` into `seed.blob` by `seed_open` (`vk_shader.cpp`). Held for the whole of `prepare_run`, then `seed_close` drops it. A non-release build does not call `prepare_run` unless `RX_VK_PRECOMPILE=1`. |
| `hookshot_programs.bin` | 5.8 GB | Append-only after a header, one record per translated variant (`program_cache_save`: `fwrite` + `fflush` under `disk_mu`). Low-mem maps it `MAP_PRIVATE`, builds an index, then `POSIX_MADV_DONTNEED` on each payload. The index (`disk_records`) is small. The bytes that hurt are the dirty page cache of the writes, and any payload faulted back in when a draw misses. |
| `mkw_vk_pipelines.bin` | 77 MB | `vkGetPipelineCacheData` into a heap vector, write to `*.tmp`, replace (`pipeline_cache_save`, every 300 presents once 32 pipelines are new). One extra copy of 77 MB for the duration of the save. The driver's live cache is larger than the file; *estimate* 150–400 MB once Kokiri's pipelines exist. |
| `hookshot_pipelines.bin` | 27 MB | Notes (`PipeNote` + an `NvnState`) appended, then read back whole by `precompile_run`. |

`disk_init` runs once (`std::call_once`). The only truncate is `fopen(..., "wb")` when the magic or `disk_stamp()` does not match. The stamp mixes the translator id, depth mode, `sizeof(Shader::Info)`, the C++ library, and every environment variable whose name starts with `RX_VK_OOT`, `RX_VK_MMA`, `RX_VK_NATIVE`, `RX_VK_NAN`, `RX_VK_TASK_CAP`, `RX_VK_NETWORK`, `RX_VK_GP`, `RX_VK_FOLLOW`, `RX_VK_NO_FOLLOW`, `RX_VK_NO_F16`, `RX_VK_SHARED`, `RX_VK_MESH`, or `EDEN_`. A Deck wrapper that exports one of those, or a crash-restart that changes one, throws the 5.8 GB file away and writes it again. That is the code path that matches "rewritten whole twice" across a kill and a relaunch. A single process does not open the file with `"wb"` twice.

A first run still writes the whole 5.8 GB once, through `fflush` on every record, into the same RAM the game is trying to use. Dirty pages are not in `MemAvailable`. On SteamOS's btrfs home partition a multi-gigabyte write is also copy-on-write, so the disk traffic can exceed the file size. `program_cache_save` does not `posix_fadvise(DONTNEED)` the range it just wrote. The low-mem `DONTNEED` only runs when the file is mapped at the next `disk_init`.

The workers that translate are the two `mkw_vk` threads. Each translation holds the Maxwell IR and the SPIR-V for one program. A mesh program's IR is the large one. *Estimate* 150–400 MB per worker while a foliage program is in flight, gone when the worker finishes. The peak at two minutes is these heaps plus the first scene's textures plus the dirty program-cache pages, not a leak that is still there at the title.

**Measure:** `/proc/self/io` `write_bytes` delta against the file's size (a ratio near 2 confirms a second full write). smaps of the `hookshot_programs.bin` VMA (`Rss`, `Pss`) and `/proc/meminfo` `Dirty` while it grows. `mincore` on that mapping, sampled, to see whether `DONTNEED` stuck.

### Audio, video, threads

- Audio (`host/nn_audio.c`): 512 voice slots, mixes, a 5 ms render. The samples live in guest memory. Host-side state is a few megabytes. Not a target.
- The intro movie (`host/nn_video.c`): NV12 at the movie's resolution, `malloc`'d per queued picture, and copied into a guest pool. 1080p NV12 is about 3 MB a frame. The queue can skip ahead (`guard < 600` is a bound on the catch-up loop, not the queue length). *Estimate* under 100 MB. It overlaps the two-minute peak and should show up as a step down in the guest smaps when the movie ends. If that step is a gigabyte, the movie is keeping RGBA too (`pool_rgba`); the code tries not to.
- Host stacks: `CreateThread(..., 64 << 20, ...)` for the game thread and for every guest thread (`hos_thread.c`, `hos_main.c`). `pthread_attr_setstacksize` reserves that much. glibc demand-pages it. Deep lifted calls touch a few megabytes, not 64. Virtual only, unless something prefaults. Fibers (`CreateFiberEx`) reserve at least 1 MB with `MAP_NORESERVE`.
- The two compiler threads are the ones that matter, and they matter as peak heap, not as stacks.

**Measure:** `smaps` entries marked `[stack]`, summed. `/proc/self/status` `Threads`. A log line when a movie decoder allocates.

## 2. Ranked work

Savings are against the ~8 GB live footprint, and they overlap. Adding them overshoots what one change will actually free. The order is the order to land them. Effort and risk assume the change stays inside the existing `RX_VK_LOW_*` / `RX_VK_DECK` switches and does not change the default on a discrete GPU.

### 1. Stop telling the game it has a Switch 2's 9809 MB

**Save:** *estimate* 1.5–3 GB of guest heap once an outdoor scene has streamed, if OoT sizes that heap from `QueryMemoryInfo`. This is the largest single lever and the least proven; the instrumentation in section 4 confirms it before any other guesswork.

**Work:** on the Deck profile, default `HOS_MEMORY_MB` to 3072 (or 2560 if the title at 3.3 GB RSS already includes a fat heap). Log the value `SetMemoryHeapSize` actually requests. Keep `MADV_DONTNEED` on free. Do not prefault (`HOS_PREFAULT` must stay unset).

**Risk:** medium. The game may allocate a fixed table out of that total and fault if the number is too small, or it may thrash its own archive streaming and hitch. 3072 is a starting point, not a measured minimum. Ship it behind the Deck profile so a desktop GPU still sees 9809.

### 2. One hard budget for textures, render targets, and staging, and evict the old scene before decoding the new one

**Save:** *estimate* 400–900 MB at the door, which is the moment earlyoom fires. Steady state inside 1280 MB may already be true; the kill is the overlap.

**Work:**

- A single ceiling, default 896 MB on the Deck, covering `total_bytes` **and** staging in flight. The current 1280 MB ignores staging until the frame ends.
- Before a wave of creates (heap jump, or `total_bytes` rising more than 64 MB in a frame), evict idle textures even if they were used inside 30 frames, as long as they were not used in the last 2. The 30-frame hold is why the house and the forest are resident together.
- Count render targets in the same ceiling. A dynamic-resolution set the GPU has not drawn for 120 frames goes, including ones that were not overwritten. 50% resolution is already selected on the Deck; the leftover 100% targets from the title and the movie should not survive into the forest.
- Run the trim from the allocator when the ceiling would break, not only from `texture_frame`.

**Risk:** medium. The file already documents the failures this causes when it is wrong: a baked LUT evicted and recreated empty (rainbow sky, black pause grade), a terrain tile recreated from stale memory, a one-frame black ground. Keep the existing exceptions for small baked targets (`baked`, under 4 MB, written once) and for anything `last_value` still in flight. The change is when the sweep runs, not which of those exceptions exist.

### 3. BC3 for ASTC on this APU, including the finer textures

**Save:** *estimate* 300–700 MB inside the texture budget, which then lets the ceiling in item 2 sit at 896 MB without dumping the scene. A 6x6 ASTC texture stored as RGBA8 is about 9 times the guest data; BC3 is about twice the guest data. The comment on the 4 GB laptop (Goron City, 1.75 GB of decoded ASTC) is the same shape as Kokiri, on a bigger card.

**Work:** treat "UMA, or driver budget over half of system RAM" like the existing `allowed < 6 GB` case, so `RX_VK_ASTC_BC_ALL` is on for the Deck profile. Lower the fine-texture floor from 64 MB to the same 2 MB as the coarse one. Leave render targets and anything the GPU writes uncompressed (`gpu_written`).

**Risk:** medium, visual. BC3 on characters and UI. Reversible with `RX_VK_ASTC_BC_ALL=0`. Do not BC-compress a texture whose top mip was already dropped and that has a single layer; the comment says those copies are blitted into level 0, which BC3 cannot take.

### 4. The 5.8 GB program cache is written at most once, and never kept

**Save:** *estimate* 1–2 GB of otherwise-dirty RAM during the first run, and the swap those dirty pages push the guest heap into. Steady state after a warm cache is already supposed to be "index only"; this item makes the first run match that, and removes the second full write.

**Work:**

- After each `fflush`, `posix_fadvise(POSIX_FADV_DONTNEED)` on the record just written. Do not mmap the file on the run that is creating it.
- Stamp check fails: rename the old file aside and start a new one. Do not rewrite in place on the same run, and do not let a Deck wrapper's `RX_VK_MESH*` / `RX_VK_OOT*` variable into the stamp unless it actually changes the SPIR-V. `RX_VK_DECK` is already not in the list; keep new Deck variables out of it.
- One compiler thread on the Deck (`vk_jobs.cpp` clamp of 2 becomes 1 when `low_mem_mode()`). *Estimate* 150–400 MB off the two-minute peak, at the cost of longer one-time stalls. The machine was 75% idle later because it was swapping, not because it needed two ACO compiles.
- Do not call `seed_open` during play. The 512 MB blob is only for `prepare_run`. A release build should run that prepare once, offline or on a first start that does nothing else, then never again.
- `pipeline_cache_save` already writes a temp file. Cap how often it runs on the Deck (the 300-present save is fine; do not also save on every scene change) and `DONTNEED` the temp buffer's pages after the rename.

**Risk:** low for the fadvise and the thread count. Medium for narrowing the stamp: a cache written with the wrong mesh flags will crash or draw garbage until it is rebuilt. Log the stamp inputs on the line that already prints `program cache %s`.

### 5. Do not pin the previous scene's pools across the load

**Save:** *estimate* 300–800 MB at the door, 0 MB once the old imports have already been reaped. The 1.3 GB Kokiri figure is live data the GPU reads. It cannot be unpinned while those draws run. The waste is the house's imports still pinned when the forest's are created, and import chunks the GPU has not read.

**Work:** on a heap jump or a pool-table generation change, reap `dead_imports` that are past `mem_completed` before `import_pool` of the new range. The reaper exists; it has to run at the start of the load, not only when a later submit notices. Keep the "GPU still in flight" rule that the comment says caused device losses (issues 132 and 195). On the Deck, do not prefetch neighbor chunks (`RX_VK_PREFETCH_CHUNKS` is already 0). Do not raise the 8 MB chunk.

**Risk:** high if a reap races a submit. The existing `value <= mem_completed` check is the one to keep. This item is ordering, not a new free path.

### 6. Romfs reads drop out of the page cache

**Save:** *estimate* 200–600 MB of `MemAvailable`, mostly during loads. Not RSS.

**Work:** `posix_fadvise(POSIX_FADV_DONTNEED)` on a file after a sequential read completes, and `POSIX_FADV_RANDOM` on open so the kernel does not readahead a 26 GB tree. Do not add a userspace cache.

**Risk:** low. The next read of the same file hits the disk. Scene loads get slightly longer and stop pushing the heap into swap.

### 7. Our own ceiling, so earlyoom is not the allocator

**Save:** none by itself. It is what keeps items 1–6 under the line that produced `mem 3.40%, swap 5.79%`.

**Work:** a process budget of **5.5 GB** counted as smaps `Rss + Swap` of the whole process, plus `Unevictable` if it is not already inside that RSS. When the next texture, import, or heap touch would pass it, run item 2's eviction and item 5's reap first, and skip the upload rather than faulting. Log once when it trips. Do not SIGTERM ourselves; earlyoom is already doing that.

**Risk:** low if the only response is eviction that already has a correct "do not evict" list. High if a tripped ceiling refuses a guest page the game needs; that is a guest OOM (`ResultOutOfMemory` from `SetMemoryHeapSize`) and is preferable to earlyoom only when the game handles it. Prefer evicting our copies first.

### Not worth doing for memory

- **Dropping `A64_WRITE_TRACK`.** The bitmap is 1 MB. The atomics cost CPU on every guest store, and this run was 75% idle. Captures are already capped at 96 MB. Revisit only if a profile shows the mark stores hot after the machine is no longer swapping.
- **Shrinking the 512 GB map, or turning off `MAP_NORESERVE`.** The reservation is free. Removing `MAP_NORESERVE` would charge the commit limit for the whole window and die at startup.
- **zram as the fix.** zram stores compressed anonymous pages in RAM, so it spends the pool we are trying to free. The guest heap has long zero runs and will compress, and a 1–2 GB zram device can be a better home for cold guest pages than the Deck's swap file (less SSD traffic than the 2 MB/s thrash). It does not raise the ceiling. Add it only after items 1–4, and only if smaps still shows cold guest `Swap` while `MemAvailable` is healthy. Do not enlarge the swap file; a bigger swap file is how this run reached 8 GB and 1 fps before earlyoom.
- **Host stack 64 MB to 8 MB.** Virtual, demand-paged. The lifted code nests a host frame per guest call (`hos_thread.c`). Leave the reservation.
- **Mirrors.** Already off. Leave them off.

### Kokiri Forest, best hypothesis

The door is the first time three waves land on a machine whose swap is already full.

1. **The swap was filled before the door.** Title RSS is 3.3 GB. About two minutes in, RSS hits 6.8 GB. That lines up with the intro movie, the first 3D scene (Link's house), and the two `mkw_vk` workers translating the programs that scene creates, while `hookshot_programs.bin` is being `fflush`'d to 5.8 GB. Dirty cache of that write is not reclaimable. The kernel swaps anonymous guest pages out to make room. RSS later reads 4–5 GB because those pages now live in swap, not because they were freed. Swap free stays at 0.
2. **The forest allocates a new pinned set before the old one is gone.** `vk_memory.cpp` records about 1.3 GB of userptr imports in Kokiri. Those pages cannot be swapped for the life of the import. The house's textures are inside the 30-frame LRU window, so they stay. Staging for the new ASTC set grows to the size of the uploads and is only trimmed at frame end. Render targets from the higher resolution used before the Deck's 50% override may still be in the 1280 MB.
3. **earlyoom fires on the overlap, not on the steady forest.** `VmRSS 4188 MiB` is what was still in RAM. The rest of the 8 GB was already in swap, and the new pinned imports could not be the pages that went there. `MemAvailable` at 3.4% and swap free at 5.8% is the kill condition. The 1–2 fps walk, had it survived, is the same working set being faulted back at 2 MB/s.

What would falsify this: a smaps sample at the door showing the guest VMA's `Rss + Swap` under 2 GB, and `Dirty` under 200 MB. Then the spike is somewhere else (a single VMA image block, a leaked import, the 512 MB seed blob held into play) and the log lines in section 4 will name it.

## 3. Target budget

Steady state while walking Kokiri, **under 6 GB** including GPU memory. The 6 GB is resident + swapped + pinned of this process, plus the dirty page cache this process is responsible for. Steam, gamescope, and the compositor need the rest of the 11.7 GB, and about 1.5 GB of `MemAvailable` has to remain so earlyoom's 10% line is not crossed.

| Category | Target | Notes |
|---|---|---|
| Guest image, resident | 250 MB | File is larger; cold pages reclaim. Do not `mlock` it. |
| Guest heap + regions, resident or swapped | 2200 MB | Requires item 1. Includes the 1.3 GB Kokiri imports, which are these same pages, pinned. |
| Lifted executable, resident | 250 MB | File-backed, reclaimable. |
| Textures + render targets | 896 MB | Item 2 and item 3. Device memory, pinned. |
| Staging + captures | 80 MB | 32 MB staging in flight, 48 MB captures. |
| Driver pipelines + our program objects | 350 MB | Warm set for the scene, not the 5.8 GB file. |
| Program-cache page cache, dirty or mapped | 64 MB | Index and the one record being loaded. |
| Audio, video, host stacks, generation tables | 80 MB | |
| **Total** | **about 4.2 GB** | Leaves about 1.5 GB of the 6 GB for the load overlap (old scene + new staging) before eviction catches up. |

If item 1 shows that the game needs more than 2200 MB of heap to get out the door without a guest OOM, lower the texture ceiling before raising the total. Do not raise the total past 6 GB.

## 4. Instrumentation

The memory line in `texture_frame` runs every 120 frames in low-mem mode and prints RSS, peak RSS, private bytes, `MemAvailable`, swap free, GPU used/allowed, staging, and `total_bytes`. It missed the kill: 120 frames at 1 fps is two minutes, and it prints staging **after** the trim. Extend that line. Do not add a new thread.

Every 30 frames, and on the same frame `MemAvailable` drops under 1500 MB, log one line:

```
mem: rss=… pss=… swap=… locked=… dirty=… avail=… swapfree=…
    guest_rss=… guest_swap=… guest_locked=… heap_mb=… image_mb=…
    tex=… tex_peak=… staging=… staging_peak=… cap=… imports=… import_mb=…
    write_bytes=… programs_mb=… workers=… dirty_cache=…
```

Where to read each field:

| Field | Source |
|---|---|
| `rss`, `pss`, `swap`, `locked` | `/proc/self/smaps_rollup` (`Rss`, `Pss`, `Swap`, `Locked`) |
| `guest_*` | The smaps entry whose range starts at `A64_HOST_BASE`. One VMA. `Rss`, `Swap`, `Locked`. |
| `heap_mb` | `hos_heap_size` and `hos_heap_used`, already in `hos_mem.c` |
| `image_mb` | The size already logged at boot |
| `tex`, `staging` | `total_bytes`, sum of `arenas[].size` **before** the trim. Keep a high-water mark since the previous line (`tex_peak`, `staging_peak`). |
| `cap` | `cap_bytes` |
| `imports`, `import_mb` | `imported.size()` and the sum of `hi - lo` |
| `write_bytes` | `/proc/self/io` `write_bytes`, as a delta |
| `programs_mb` | file size, plus smaps `Rss` of that mapping if it is mapped |
| `dirty_cache` | `/proc/meminfo` `Dirty` + `Writeback` |
| `workers` | `pending` in `vk_jobs.cpp` |
| `avail`, `swapfree` | already read |

`smaps_rollup` is one page and is cheap enough every 30 frames. The per-VMA scan is not: the guest window is one VMA, so find that line and the executable line and stop. Do not walk all 186k file mappings.

Once, on the way out the door (first time the guest heap grows by more than 64 MB after the title, or the first time `import_mb` crosses 512), log the same line with a `scene=kokiri` tag and dump `vma_log_heaps`. That is the sample that confirms or kills the hypothesis in section 2.

Also log, once, at `disk_init`: stamp, fopen mode (`wb` versus `ab`), file size before, and whether the map was kept. A second `wb` in one session is the rewrite bug; a second process start with a different stamp is the "twice" the Deck run saw.

`HOS_MEM_MAP` stays off. It cannot see Linux residency.

## 5. White Link

The shot is the wake-up in the house: the pose is a full-size body on the bed, arm out, and the bed, straw, and wall are lit. The body is a flat glowing white with a bloom halo.

What the code says that is **not**:

- Bones read as zero collapse a skinned mesh into spikes at the origin (`vk_bank.cpp`, the pointer-table comment). This silhouette is the right size and in the right place, so the bone pointers are not the zero-read failure.
- The per-character shadow bias (`vk_draw.cpp`, the 768² depth map, program `1e26755d`) turns Link **black** in the shade when it is wrong. This frame is the opposite.
- A missing texture on RADV is a null descriptor when `nullDescriptor` is set. `make_dummies` returns without building a stand-in in that case (`vk_draw.cpp`). A null sample is (0,0,0,0), which is black or invisible, not a white glow, unless this material does `1 - sample` or adds a white vertex color on top of a zero albedo. Check `st_dummy_textures` and the `sampled as the dummy` log during this shot before chasing anything else. If those fire for Link's handles, the bug is the unresolved handle, and the white is the shader's response to zero.

What the code says is the likely cause:

Link is the skinned mesh. The room is ordinary geometry. Mesh outputs are built in `DefineMesh` (`spirv_emit_context.cpp`) as **one `vec4` of float per generic location**, and `CopyLines` bitcasts every stored word to float. A vertex-shader output goes through `DefineGenericOutput`, which can be a narrower vector. Position is a separate builtin, so a mesh program can export a correct `gl_Position` and garbage varyings in the same draw. Garbage UVs with a sampler whose border is opaque white (`VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE`, the path in `vk_draw.cpp` that matches border `(1,1,1,1)`) sample white. The bloom chain then does what `docs/design.md` already recorded for a single hot pixel: it spreads. A flat white mesh becomes a glowing silhouette, and the room, drawn with shaders whose UVs match, does not.

The same function forces every generic through a float output. If the fragment stage was compiled against integer or narrower mesh outputs (packed normals, a material id), RADV will interpolate the wrong bits and the lighting term saturates. That also reads as white, with the same "position is fine" signature.

This branch turned the driver's optimizer **on** for RADV task/mesh programs (`vk_draw.cpp`, `RX_VK_MESH_OPT`, the comment that unoptimized mesh shaders never finish on RADV). An optimizer that drops a varying the fragment still reads produces this picture and would not show up on a desktop AMDGPU path that still builds those pipelines with `DISABLE_OPTIMIZATION`. Compare one frame with `RX_VK_MESH_OPT=0` only if the draw is actually a mesh draw; on RADV that flag hangs the gfx ring, so use it as a capture of a single frame, not as a play session.

How to confirm, without a new theory:

1. The draw log for this frame (`RX_VK_RECORD` / the pass trace). If Link's draw is a mesh or task program, record its key and whether the fragment inputs are generic floats.
2. `st_dummy_textures` over those presents. Nonzero means the null-descriptor path, and the mesh-varying theory is the wrong one.
3. If the mesh key is known, dump that one program (`RX_VK_DUMP_STAGES` is per stage) and compare output locations to the fragment inputs. A `vec4` float mesh output bound to a fragment input that is not four floats is the bug.

The dream line of dialogue is not, by itself, an explanation. If he is still flat white after he stands up and the tunic should be green, it is the mesh interface. If only this shot is white and the next gameplay frame is textured, it is a material of the dream pass (an emissive that the exposure scale in `oot_visual_workaround.h` would have pulled down, and that workaround is off unless `RX_VK_OOT_VISUAL=1`). Those are distinguishable in two frames of the same capture.

## Order to land it

1. Section 4's log line, and one trip out the door. Confirm guest `Rss+Swap`, `Dirty`, and `import_mb`.
2. Item 4 (program cache write + one compiler thread). Stops the first-run swap fill even if the heap guess is wrong.
3. Item 1 (`HOS_MEMORY_MB`) once the log shows the heap is the guest cost.
4. Items 2 and 3 together (ceiling, then BC3 so the ceiling still holds the scene).
5. Item 5 (reap imports before the new scene) if the door sample still shows `import_mb` above the new scene's own pools.
6. Item 6 and item 7 as the safety net, in the same change as the ceiling.

Desktop GPUs stay on the current defaults. Every cut is a `low_mem_on(...)` switch that can be turned off with `=0`.
