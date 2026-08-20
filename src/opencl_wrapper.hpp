#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

// Standard OpenCL Types & Constants
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef size_t cl_device_info;
typedef unsigned long long cl_mem_flags;
typedef unsigned long long cl_command_queue_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004

#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MEM_COPY_HOST_PTR (1 << 5)

#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_TRUE 1
#define CL_FALSE 0

typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (*pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (*pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context);

class OpenCLContext {
public:
    bool available = false;
    std::string device_name = "None";
    int compute_units = 0;
    size_t global_vram_mb = 0;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    pfn_clGetPlatformIDs clGetPlatformIDs = nullptr;
    pfn_clGetDeviceIDs clGetDeviceIDs = nullptr;
    pfn_clGetDeviceInfo clGetDeviceInfo = nullptr;
    pfn_clCreateContext clCreateContext = nullptr;
    pfn_clCreateCommandQueue clCreateCommandQueue = nullptr;
    pfn_clCreateBuffer clCreateBuffer = nullptr;
    pfn_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
    pfn_clBuildProgram clBuildProgram = nullptr;
    pfn_clGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
    pfn_clCreateKernel clCreateKernel = nullptr;
    pfn_clSetKernelArg clSetKernelArg = nullptr;
    pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
    pfn_clEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
    pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
    pfn_clFinish clFinish = nullptr;
    pfn_clReleaseMemObject clReleaseMemObject = nullptr;
    pfn_clReleaseKernel clReleaseKernel = nullptr;
    pfn_clReleaseProgram clReleaseProgram = nullptr;
    pfn_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
    pfn_clReleaseContext clReleaseContext = nullptr;

private:
#ifdef _WIN32
    HMODULE hLib = nullptr;
#else
    void* hLib = nullptr;
#endif

    void* load_sym(const char* name) {
#ifdef _WIN32
        return (void*)GetProcAddress(hLib, name);
#else
        return dlsym(hLib, name);
#endif
    }

public:
    OpenCLContext() {
#ifdef _WIN32
        hLib = LoadLibraryA("OpenCL.dll");
#else
        hLib = dlopen("libOpenCL.so", RTLD_LAZY);
        if (!hLib) hLib = dlopen("libOpenCL.so.1", RTLD_LAZY);
#endif
        if (!hLib) return;

        clGetPlatformIDs = (pfn_clGetPlatformIDs)load_sym("clGetPlatformIDs");
        clGetDeviceIDs = (pfn_clGetDeviceIDs)load_sym("clGetDeviceIDs");
        clGetDeviceInfo = (pfn_clGetDeviceInfo)load_sym("clGetDeviceInfo");
        clCreateContext = (pfn_clCreateContext)load_sym("clCreateContext");
        clCreateCommandQueue = (pfn_clCreateCommandQueue)load_sym("clCreateCommandQueue");
        clCreateBuffer = (pfn_clCreateBuffer)load_sym("clCreateBuffer");
        clCreateProgramWithSource = (pfn_clCreateProgramWithSource)load_sym("clCreateProgramWithSource");
        clBuildProgram = (pfn_clBuildProgram)load_sym("clBuildProgram");
        clGetProgramBuildInfo = (pfn_clGetProgramBuildInfo)load_sym("clGetProgramBuildInfo");
        clCreateKernel = (pfn_clCreateKernel)load_sym("clCreateKernel");
        clSetKernelArg = (pfn_clSetKernelArg)load_sym("clSetKernelArg");
        clEnqueueNDRangeKernel = (pfn_clEnqueueNDRangeKernel)load_sym("clEnqueueNDRangeKernel");
        clEnqueueReadBuffer = (pfn_clEnqueueReadBuffer)load_sym("clEnqueueReadBuffer");
        clEnqueueWriteBuffer = (pfn_clEnqueueWriteBuffer)load_sym("clEnqueueWriteBuffer");
        clFinish = (pfn_clFinish)load_sym("clFinish");
        clReleaseMemObject = (pfn_clReleaseMemObject)load_sym("clReleaseMemObject");
        clReleaseKernel = (pfn_clReleaseKernel)load_sym("clReleaseKernel");
        clReleaseProgram = (pfn_clReleaseProgram)load_sym("clReleaseProgram");
        clReleaseCommandQueue = (pfn_clReleaseCommandQueue)load_sym("clReleaseCommandQueue");
        clReleaseContext = (pfn_clReleaseContext)load_sym("clReleaseContext");

        if (!clGetPlatformIDs || !clGetDeviceIDs || !clCreateContext) return;

        cl_uint num_platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) return;

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        for (cl_uint p = 0; p < num_platforms; ++p) {
            cl_uint num_devices = 0;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

                platform = platforms[p];
                device = devices[0]; // Select primary GPU

                char name[256] = {0};
                cl_uint cu = 0;
                cl_ulong gmem = 0;

                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
                clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);

                device_name = name;
                compute_units = cu;
                global_vram_mb = gmem / (1024 * 1024);

                cl_int err = 0;
                context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
                if (err == CL_SUCCESS && context) {
                    queue = clCreateCommandQueue(context, device, 0, &err);
                    if (err == CL_SUCCESS && queue) {
                        available = true;
                        break;
                    }
                }
            }
        }
    }

    ~OpenCLContext() {
        if (queue && clReleaseCommandQueue) clReleaseCommandQueue(queue);
        if (context && clReleaseContext) clReleaseContext(context);
#ifdef _WIN32
        if (hLib) FreeLibrary(hLib);
#else
        if (hLib) dlclose(hLib);
#endif
    }
};
