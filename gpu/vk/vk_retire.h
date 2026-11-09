#pragma once

// Destroying an image that has no CPU copy brings it back cleared (the white foliage).
// The working-set pass (ws) and a scene purge (scene) both leave those images resident.
// Desktop is neither of those: the older time rules still decide, including the course arrays
// a desktop card retires after they have been unused. Pressure does not override this.
inline bool keep_image_that_would_clear(bool ws, bool scene, bool overwritten, bool gpu_written, bool released,
                                        bool evict_rendered, bool wrote_back, bool bc) {
    if (!(ws || scene) || overwritten) return false;
    if (gpu_written && !released && !evict_rendered && !wrote_back) return true;
    if (bc) return true;
    return false;
}
