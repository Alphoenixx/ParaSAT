#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <iostream>
#include <string>
#include <thread>
#include <algorithm>

struct SystemHardwareSpecs {
    // CPU
    int total_cpu_threads = 1;
    int default_cpu_threads = 1;

    // GPU (CUDA)
    bool cuda_available = false;
    std::string gpu_name = "None";
    int sm_count = 0;
    int total_cuda_cores = 0;
    int default_cuda_cores = 0;
    size_t vram_mb = 0;
    int compute_major = 0;
    int compute_minor = 0;

    void print_summary() const {
        std::cout << "========================================================================\n";
        std::cout << "                   PARASAT HARDWARE AUTO-DETECTION                      \n";
        std::cout << "========================================================================\n";
        std::cout << "CPU Hardware Threads:   " << total_cpu_threads << " (Default: " << default_cpu_threads << " threads / 50% allocation)\n";
        if (cuda_available) {
            std::cout << "NVIDIA GPU Detected:    " << gpu_name << " (Compute " << compute_major << "." << compute_minor << ")\n";
            std::cout << "Streaming Multiprocess: " << sm_count << " SMs (" << total_cuda_cores << " CUDA Cores)\n";
            std::cout << "Default CUDA Cores:     " << default_cuda_cores << " Cores (50% allocation)\n";
            std::cout << "Dedicated GPU VRAM:     " << vram_mb << " MB\n";
        } else {
            std::cout << "GPU Acceleration:       Not Detected / CPU-Only Mode Active\n";
        }
        std::cout << "========================================================================\n";
    }
};

class HardwareDetector {
public:
    static SystemHardwareSpecs detect() {
        SystemHardwareSpecs specs;

        // 1. CPU Detection
        unsigned int hw_threads = std::thread::hardware_concurrency();
        specs.total_cpu_threads = (hw_threads > 0) ? static_cast<int>(hw_threads) : 4;
        specs.default_cpu_threads = std::max(1, specs.total_cpu_threads / 2);

        // 2. CUDA Driver Detection (via nvcuda.dll / libcuda.so)
#ifdef _WIN32
        HMODULE hCuda = LoadLibraryA("nvcuda.dll");
#else
        void* hCuda = dlopen("libcuda.so", RTLD_LAZY);
        if (!hCuda) hCuda = dlopen("libcuda.so.1", RTLD_LAZY);
#endif
        if (hCuda) {
            typedef int (*pfn_cuInit)(unsigned int);
            typedef int (*pfn_cuDeviceGetCount)(int*);
            typedef int (*pfn_cuDeviceGet)(int*, int);
            typedef int (*pfn_cuDeviceGetName)(char*, int, int);
            typedef int (*pfn_cuDeviceGetAttribute)(int*, int, int);
            typedef int (*pfn_cuDeviceTotalMem)(size_t*, int);

#ifdef _WIN32
            auto cuInit = (pfn_cuInit)GetProcAddress(hCuda, "cuInit");
            auto cuDeviceGetCount = (pfn_cuDeviceGetCount)GetProcAddress(hCuda, "cuDeviceGetCount");
            auto cuDeviceGet = (pfn_cuDeviceGet)GetProcAddress(hCuda, "cuDeviceGet");
            auto cuDeviceGetName = (pfn_cuDeviceGetName)GetProcAddress(hCuda, "cuDeviceGetName");
            auto cuDeviceGetAttribute = (pfn_cuDeviceGetAttribute)GetProcAddress(hCuda, "cuDeviceGetAttribute");
            auto cuDeviceTotalMem = (pfn_cuDeviceTotalMem)GetProcAddress(hCuda, "cuDeviceTotalMem_v2");
            if (!cuDeviceTotalMem) cuDeviceTotalMem = (pfn_cuDeviceTotalMem)GetProcAddress(hCuda, "cuDeviceTotalMem");
#else
            auto cuInit = (pfn_cuInit)dlsym(hCuda, "cuInit");
            auto cuDeviceGetCount = (pfn_cuDeviceGetCount)dlsym(hCuda, "cuDeviceGetCount");
            auto cuDeviceGet = (pfn_cuDeviceGet)dlsym(hCuda, "cuDeviceGet");
            auto cuDeviceGetName = (pfn_cuDeviceGetName)dlsym(hCuda, "cuDeviceGetName");
            auto cuDeviceGetAttribute = (pfn_cuDeviceGetAttribute)dlsym(hCuda, "cuDeviceGetAttribute");
            auto cuDeviceTotalMem = (pfn_cuDeviceTotalMem)dlsym(hCuda, "cuDeviceTotalMem_v2");
            if (!cuDeviceTotalMem) cuDeviceTotalMem = (pfn_cuDeviceTotalMem)dlsym(hCuda, "cuDeviceTotalMem");
#endif

            if (cuInit && cuDeviceGetCount && cuDeviceGet && cuInit(0) == 0) {
                int count = 0;
                if (cuDeviceGetCount(&count) == 0 && count > 0) {
                    int dev = 0;
                    if (cuDeviceGet(&dev, 0) == 0) {
                        char name[256] = {0};
                        cuDeviceGetName(name, 256, dev);
                        specs.gpu_name = name;

                        int sm = 0, major = 0, minor = 0;
                        cuDeviceGetAttribute(&sm, 16, dev);     // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
                        cuDeviceGetAttribute(&major, 75, dev);  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
                        cuDeviceGetAttribute(&minor, 76, dev);  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

                        specs.sm_count = sm;
                        specs.compute_major = major;
                        specs.compute_minor = minor;

                        int cores_per_sm = (major == 7) ? 64 : ((major >= 8) ? 128 : 64);
                        specs.total_cuda_cores = sm * cores_per_sm;
                        specs.default_cuda_cores = std::max(32, specs.total_cuda_cores / 2);

                        if (cuDeviceTotalMem) {
                            size_t bytes = 0;
                            cuDeviceTotalMem(&bytes, dev);
                            specs.vram_mb = bytes / (1024 * 1024);
                        }

                        specs.cuda_available = true;
                    }
                }
            }

#ifdef _WIN32
            FreeLibrary(hCuda);
#else
            dlclose(hCuda);
#endif
        }

        return specs;
    }
};
