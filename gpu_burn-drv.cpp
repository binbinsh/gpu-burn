/*
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 *those of the authors and should not be interpreted as representing official
 *policies, either expressed or implied, of the FreeBSD Project.
 */

// Matrices are SIZE*SIZE..  POT should be efficiently implemented in CUBLAS
#define SIZE 8192ul
#define USEMEM 0.9 // Try to allocate 90% of memory
#define COMPARE_KERNEL "compare.ptx"

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
//#define OPS_PER_MUL 17188257792ul // Measured for SIZE = 2048
#define OPS_PER_MUL 1100048498688ul // Extrapolated for SIZE = 8192

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fstream>
#include <map>
#include <signal.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <regex>
#include <sstream>

#define SIGTERM_TIMEOUT_THRESHOLD_SECS 30 // number of seconds for sigterm to kill child processes before forcing a sigkill

#include "cublas_v2.h"
#define CUDA_ENABLE_DEPRECATED
#include <cuda.h>
#include <cuda_bf16.h> // BF16 CHANGE: Include bfloat16 header

void _checkError(int rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUDA_SUCCESS) {
        const char *err;
        cuGetErrorString((CUresult)rCode, &err);

        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

void _checkError(cublasStatus_t rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUBLAS_STATUS_SUCCESS) {
#if CUBLAS_VER_MAJOR >= 12
        const char *err = cublasGetStatusString(rCode);
#else
        const char *err = "";
#endif
        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

#define checkError(rCode, ...)                                                 \
    _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

bool g_running = false;

template <class T> class GPU_Test {
  public:
    // BF16 CHANGE: Add bf16 flag to constructor
    GPU_Test(int dev, bool doubles, bool tensors, bool bf16, const char *kernelFile, int nvsmiIndex = -1)
    : d_devNumber(dev), d_doubles(doubles), d_tensors(tensors), d_bf16(bf16), d_kernelFile(kernelFile), d_nvsmiIndex(nvsmiIndex){
        checkError(cuDeviceGet(&d_dev, d_devNumber));
        checkError(cuCtxCreate(&d_ctx, 0, d_dev));

        bind();

        checkError(cublasCreate(&d_cublas), "init");

        if (d_tensors)
            checkError(cublasSetMathMode(d_cublas, CUBLAS_TENSOR_OP_MATH));

        checkError(cuMemAllocHost((void **)&d_faultyElemsHost, sizeof(int)));
        d_error = 0;

        g_running = true;

        struct sigaction action;
        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = termHandler;
        sigaction(SIGTERM, &action, NULL);
    }
    ~GPU_Test() {
        bind();
        checkError(cuMemFree(d_Cdata), "Free A");
        checkError(cuMemFree(d_Adata), "Free B");
        checkError(cuMemFree(d_Bdata), "Free C");
        cuMemFreeHost(d_faultyElemsHost);
        printf("Freed memory for dev %d\n", d_devNumber);

        cublasDestroy(d_cublas);
        printf("Uninitted cublas\n");
    }

    static void termHandler(int signum) { g_running = false; }

    unsigned long long int getErrors() {
        if (*d_faultyElemsHost) {
            d_error += (long long int)*d_faultyElemsHost;
        }
        unsigned long long int tempErrs = d_error;
        d_error = 0;
        return tempErrs;
    }

    size_t getIters() { return d_iters; }

    void bind() { checkError(cuCtxSetCurrent(d_ctx), "Bind CTX"); }

    size_t totalMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return totalMem;
    }

    size_t availMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return freeMem;
    }

    void initBuffers(T *A, T *B, ssize_t useBytes = 0) {
        bind();

        if (useBytes == 0)
            useBytes = (ssize_t)((double)availMemory() * USEMEM);
        if (useBytes < 0)
            useBytes = (ssize_t)((double)availMemory() * (-useBytes / 100.0));
        
        // BF16 CHANGE: Update status message to show bfloat16 type
        printf("Initialized device %d with %lu MB of memory (%lu MB available, "
               "using %lu MB of it), %s%s\n",
               d_nvsmiIndex >= 0 ? d_nvsmiIndex : d_devNumber,
               totalMemory() / 1024ul / 1024ul,
               availMemory() / 1024ul / 1024ul, useBytes / 1024ul / 1024ul,
               d_doubles ? "using DOUBLES" : (d_bf16 ? "using BFLOAT16" : "using FLOATS"),
               d_tensors ? ", using Tensor Cores" : "");
        d_resultSize = sizeof(T) * SIZE * SIZE;
        d_iters = (useBytes - 2 * d_resultSize) /
                  d_resultSize; // We remove A and B sizes
        printf("Results are %zu bytes each, thus performing %zu iterations\n",
               d_resultSize, d_iters);
        if ((size_t)useBytes < 3 * d_resultSize)
            throw std::string("Low mem for result. aborting.\n");
        checkError(cuMemAlloc(&d_Cdata, d_iters * d_resultSize), "C alloc");
        checkError(cuMemAlloc(&d_Adata, d_resultSize), "A alloc");
        checkError(cuMemAlloc(&d_Bdata, d_resultSize), "B alloc");

        checkError(cuMemAlloc(&d_faultyElemData, sizeof(int)), "faulty data");

        // Populating matrices A and B
        checkError(cuMemcpyHtoD(d_Adata, A, d_resultSize), "A -> device");
        checkError(cuMemcpyHtoD(d_Bdata, B, d_resultSize), "B -> device");

        initCompareKernel();
    }

    void compute() {
        bind();
        static const float alpha = 1.0f;
        static const float beta = 0.0f;
        static const double alphaD = 1.0;
        static const double betaD = 0.0;

        for (size_t i = 0; i < d_iters; ++i) {
            if (d_doubles)
                checkError(
                    cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,
                                SIZE, &alphaD, (const double *)d_Adata, SIZE,
                                (const double *)d_Bdata, SIZE, &betaD,
                                (double *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "DGEMM");
            // BF16 CHANGE: Add logic for bfloat16 GEMM using cublasGemmEx
            else if (d_bf16) {
                cublasGemmAlgo_t algo = d_tensors ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
                checkError(
                    cublasGemmEx(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,
                                SIZE, &alpha, (const void*)d_Adata, CUDA_R_16BF, SIZE,
                                (const void*)d_Bdata, CUDA_R_16BF, SIZE, &beta,
                                (__nv_bfloat16 *)d_Cdata + i * SIZE * SIZE, CUDA_R_16BF, SIZE,
                                CUDA_R_32F, algo),
                    "HGEMM (BF16)");
            }
            else
                checkError(
                    cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,
                                SIZE, &alpha, (const float *)d_Adata, SIZE,
                                (const float *)d_Bdata, SIZE, &beta,
                                (float *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "SGEMM");
        }
    }

    void initCompareKernel() {
        {
            std::ifstream f(d_kernelFile);
            checkError(f.good() ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND,
                       std::string("couldn't find compare kernel: ") + d_kernelFile);
        }
        checkError(cuModuleLoad(&d_module, d_kernelFile), "load module");
        
        // BF16 CHANGE: Select the correct kernel name based on the data type
        const char* kernel_name;
        if (d_doubles) {
            kernel_name = "compareD";
        } else if (d_bf16) {
            kernel_name = "compareBf16";
        } else {
            kernel_name = "compare";
        }
        checkError(cuModuleGetFunction(&d_function, d_module, kernel_name), "get func");

        checkError(cuFuncSetCacheConfig(d_function, CU_FUNC_CACHE_PREFER_L1),
                   "L1 config");
        checkError(cuParamSetSize(d_function, __alignof(T *) +
                                                  __alignof(int *) +
                                                  __alignof(size_t)),
                   "set param size");
        checkError(cuParamSetv(d_function, 0, &d_Cdata, sizeof(T *)),
                   "set param");
        checkError(cuParamSetv(d_function, __alignof(T *), &d_faultyElemData,
                               sizeof(T *)),
                   "set param");
        checkError(cuParamSetv(d_function, __alignof(T *) + __alignof(int *),
                               &d_iters, sizeof(size_t)),
                   "set param");

        checkError(cuFuncSetBlockShape(d_function, g_blockSize, g_blockSize, 1),
                   "set block size");
    }

    void compare() {
        checkError(cuMemsetD32Async(d_faultyElemData, 0, 1, 0), "memset");
        checkError(cuLaunchGridAsync(d_function, SIZE / g_blockSize,
                                     SIZE / g_blockSize, 0),
                   "Launch grid");
        checkError(cuMemcpyDtoHAsync(d_faultyElemsHost, d_faultyElemData,
                                     sizeof(int), 0),
                   "Read faultyelemdata");
    }

    bool shouldRun() { return g_running; }

  private:
    int d_nvsmiIndex;   // nvidia-smi index of the GPU
    bool d_doubles;
    bool d_tensors;
    bool d_bf16; // BF16 CHANGE: Add flag for bfloat16
    int d_devNumber;
    const char *d_kernelFile;
    size_t d_iters;
    size_t d_resultSize;

    long long int d_error;

    static const int g_blockSize = 16;

    CUdevice d_dev;
    CUcontext d_ctx;
    CUmodule d_module;
    CUfunction d_function;

    CUdeviceptr d_Cdata;
    CUdeviceptr d_Adata;
    CUdeviceptr d_Bdata;
    CUdeviceptr d_faultyElemData;
    int *d_faultyElemsHost;

    cublasHandle_t d_cublas;
};

// Returns the number of devices
int initCuda() {
    try {
        checkError(cuInit(0));
    } catch (std::runtime_error e) {
        fprintf(stderr, "Couldn't init CUDA: %s\n", e.what());
        return 0;
    }
    int deviceCount = 0;
    checkError(cuDeviceGetCount(&deviceCount));

    if (!deviceCount)
        throw std::string("No CUDA devices");

#ifdef USEDEV
    if (USEDEV >= deviceCount)
        throw std::string("Not enough devices for USEDEV");
#endif

    return deviceCount;
}

// Get GPU UUID by CUDA device index
std::string getGpuUuid(int cudaDeviceIndex) {
    CUdevice device;
    CUuuid uuid;
    checkError(cuDeviceGet(&device, cudaDeviceIndex));
    checkError(cuDeviceGetUuid(&uuid, device));
    
    char uuidStr[37]; // UUID string format: 8-4-4-4-12 (32 hex chars + 4 dashes + null terminator)
    snprintf(uuidStr, sizeof(uuidStr), 
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             (unsigned char)uuid.bytes[0], (unsigned char)uuid.bytes[1], 
             (unsigned char)uuid.bytes[2], (unsigned char)uuid.bytes[3],
             (unsigned char)uuid.bytes[4], (unsigned char)uuid.bytes[5],
             (unsigned char)uuid.bytes[6], (unsigned char)uuid.bytes[7],
             (unsigned char)uuid.bytes[8], (unsigned char)uuid.bytes[9],
             (unsigned char)uuid.bytes[10], (unsigned char)uuid.bytes[11],
             (unsigned char)uuid.bytes[12], (unsigned char)uuid.bytes[13],
             (unsigned char)uuid.bytes[14], (unsigned char)uuid.bytes[15]);
    
    return std::string("GPU-") + uuidStr;
}

// Create mapping between nvidia-smi device index and CUDA device index
// Returns a vector where vector[nvidia_smi_index] = cuda_index
std::vector<int> createNvidiaSmiToCudaMapping(int cudaDeviceCount) {
    std::vector<int> mapping;
    
#if IS_JETSON
    // On Jetson, we assume the ordering is the same
    for (int i = 0; i < cudaDeviceCount; i++) {
        mapping.push_back(i);
    }
#else
    // Get UUIDs from CUDA devices
    std::vector<std::string> cudaUuids;
    for (int i = 0; i < cudaDeviceCount; i++) {
        cudaUuids.push_back(getGpuUuid(i));
    }
    
    // Parse nvidia-smi output to get UUID ordering
    FILE *fp = popen("nvidia-smi -L", "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to run nvidia-smi -L, assuming same order as CUDA\n");
        for (int i = 0; i < cudaDeviceCount; i++) {
            mapping.push_back(i);
        }
        return mapping;
    }
    
    char line[512];
    while (fgets(line, sizeof(line), fp) != NULL) {
        // Parse lines like: "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
        char *uuidStart = strstr(line, "UUID: ");
        if (uuidStart) {
            uuidStart += 6; // Skip "UUID: "
            char *uuidEnd = strchr(uuidStart, ')');
            if (uuidEnd) {
                *uuidEnd = '\0';
                std::string nvidiaUuid(uuidStart);
                
                // Find matching CUDA device
                for (int cudaIdx = 0; cudaIdx < cudaDeviceCount; cudaIdx++) {
                    if (cudaUuids[cudaIdx] == nvidiaUuid) {
                        mapping.push_back(cudaIdx);
                        break;
                    }
                }
            }
        }
    }
    
    pclose(fp);
    
    // Validate mapping
    if (mapping.size() != (size_t)cudaDeviceCount) {
        fprintf(stderr, "Warning: Could not create complete nvidia-smi to CUDA mapping. Using default order.\n");
        mapping.clear();
        for (int i = 0; i < cudaDeviceCount; i++) {
            mapping.push_back(i);
        }
    }
#endif
    
    return mapping;
}

// BF16 CHANGE: Add bf16 flag to startBurn function
template <class T>
void startBurn(int index, int writeFd, T *A, T *B, bool doubles, bool tensors, bool bf16,
               ssize_t useBytes, const char *kernelFile, int nvsmiIndex = -1) {
    GPU_Test<T> *our;
    try {
        // BF16 CHANGE: Pass bf16 flag to GPU_Test constructor
        our = new GPU_Test<T>(index, doubles, tensors, bf16, kernelFile, nvsmiIndex);
        our->initBuffers(A, B, useBytes);
    } catch (const std::exception &e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.what());
        exit(EMEDIUMTYPE);
    }

    // The actual work
    try {
        int eventIndex = 0;
        const int maxEvents = 2;
        CUevent events[maxEvents];
        for (int i = 0; i < maxEvents; ++i)
            cuEventCreate(events + i, 0);

        int nonWorkIters = maxEvents;

        while (our->shouldRun()) {
            our->compute();
            our->compare();
            checkError(cuEventRecord(events[eventIndex], 0), "Record event");

            eventIndex = ++eventIndex % maxEvents;

            while (cuEventQuery(events[eventIndex]) != CUDA_SUCCESS)
                usleep(1000);

            if (--nonWorkIters > 0)
                continue;

            int ops = our->getIters();
            write(writeFd, &ops, sizeof(int));
            ops = our->getErrors();
            write(writeFd, &ops, sizeof(int));
        }

        for (int i = 0; i < maxEvents; ++i)
            cuEventSynchronize(events[i]);
        delete our;
    } catch (const std::exception &e) {
        fprintf(stderr, "Failure during compute: %s\n", e.what());
        int ops = -1;
        // Signalling that we failed
        write(writeFd, &ops, sizeof(int));
        write(writeFd, &ops, sizeof(int));
        exit(ECONNREFUSED);
    }
}

int pollTemp(pid_t *p) {
    int tempPipe[2];
    pipe(tempPipe);

    pid_t myPid = fork();

    if (!myPid) {
        close(tempPipe[0]);
        dup2(tempPipe[1], STDOUT_FILENO);
#if IS_JETSON
        execlp("tegrastats", "tegrastats", "--interval", "5000", NULL);
        fprintf(stderr, "Could not invoke tegrastats, no temps available\n");
#else
        execlp("nvidia-smi", "nvidia-smi", "-l", "5", "-q", "-d", "TEMPERATURE",
               NULL);
        fprintf(stderr, "Could not invoke nvidia-smi, no temps available\n");
#endif

        exit(ENODEV);
    }

    *p = myPid;
    close(tempPipe[1]);

    return tempPipe[0];
}

void updateTemps(int handle, std::vector<int> *temps) {
    const int readSize = 10240;
    static int gpuIter = 0;
    char data[readSize + 1];

    int curPos = 0;
    do {
        read(handle, data + curPos, sizeof(char));
    } while (data[curPos++] != '\n');

    data[curPos - 1] = 0;

#if IS_JETSON
    std::string data_str(data);
    std::regex pattern("GPU@([0-9]+)C");
    std::smatch matches;
    if (std::regex_search(data_str, matches, pattern)) {
        if (matches.size() > 1) {
            int tempValue = std::stoi(matches[1]);
            temps->at(gpuIter) = tempValue;
            gpuIter = (gpuIter + 1) % (temps->size());
        }
    }
#else
    // FIXME: The syntax of this print might change in the future..
    int tempValue;
    if (sscanf(data,
               "        GPU Current Temp            : %d C",
               &tempValue) == 1) {
        temps->at(gpuIter) = tempValue;
        gpuIter = (gpuIter + 1) % (temps->size());
    } else if (!strcmp(data, "      Gpu             "
                             "   : N/A"))
        gpuIter =
            (gpuIter + 1) %
            (temps->size()); // We rotate the iterator for N/A values as well
#endif
}

void updateTempsWithMapping(int handle, std::vector<int> *temps, const std::vector<int> &nvidiaSmiToCudaMap, int specificCudaDevice = -1) {
    const int readSize = 10240;
    static int nvidiaSmiGpuIndex = 0;
    char data[readSize + 1];

    int curPos = 0;
    do {
        read(handle, data + curPos, sizeof(char));
    } while (data[curPos++] != '\n');

    data[curPos - 1] = 0;

#if IS_JETSON
    std::string data_str(data);
    std::regex pattern("GPU@([0-9]+)C");
    std::smatch matches;
    if (std::regex_search(data_str, matches, pattern)) {
        if (matches.size() > 1) {
            int tempValue = std::stoi(matches[1]);
            // Map nvidia-smi index to CUDA index
            if (nvidiaSmiGpuIndex < nvidiaSmiToCudaMap.size()) {
                int cudaIndex = nvidiaSmiToCudaMap[nvidiaSmiGpuIndex];
                if (specificCudaDevice >= 0) {
                    // Single GPU mode: only update if this is our device
                    if (cudaIndex == specificCudaDevice && temps->size() > 0) {
                        temps->at(0) = tempValue;
                    }
                } else {
                    // Multi-GPU mode: normal mapping
                    if (cudaIndex < temps->size()) {
                        temps->at(cudaIndex) = tempValue;
                    }
                }
            }
            nvidiaSmiGpuIndex = (nvidiaSmiGpuIndex + 1) % nvidiaSmiToCudaMap.size();
        }
    }
#else
    // FIXME: The syntax of this print might change in the future..
    int tempValue;
    if (sscanf(data,
               "        GPU Current Temp            : %d C",
               &tempValue) == 1) {
        // Map nvidia-smi index to CUDA index
        if (nvidiaSmiGpuIndex < nvidiaSmiToCudaMap.size()) {
            int cudaIndex = nvidiaSmiToCudaMap[nvidiaSmiGpuIndex];
            if (specificCudaDevice >= 0) {
                // Single GPU mode: only update if this is our device
                if (cudaIndex == specificCudaDevice && temps->size() > 0) {
                    temps->at(0) = tempValue;
                }
            } else {
                // Multi-GPU mode: normal mapping
                if (cudaIndex < temps->size()) {
                    temps->at(cudaIndex) = tempValue;
                }
            }
        }
        nvidiaSmiGpuIndex = (nvidiaSmiGpuIndex + 1) % nvidiaSmiToCudaMap.size();
    } else if (!strcmp(data, "      Gpu             "
                             "   : N/A")) {
        nvidiaSmiGpuIndex = (nvidiaSmiGpuIndex + 1) % nvidiaSmiToCudaMap.size();
    }
#endif
}

void listenClients(std::vector<int> clientFd, std::vector<pid_t> clientPid,
                   int runLength,
                   std::chrono::seconds sigterm_timeout_threshold_secs,
                   const std::vector<int> &nvidiaSmiToCudaMap,
                   int specificCudaDevice = -1) {
    fd_set waitHandles;

    pid_t tempPid;
    int tempHandle = pollTemp(&tempPid);
    int maxHandle = tempHandle;

    FD_ZERO(&waitHandles);
    FD_SET(tempHandle, &waitHandles);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        if (clientFd.at(i) > maxHandle)
            maxHandle = clientFd.at(i);
        FD_SET(clientFd.at(i), &waitHandles);
    }

    std::vector<int> clientTemp;
    std::vector<int> clientErrors;
    std::vector<int> clientCalcs;
    std::vector<bool> clientFaulty;

    for (size_t i = 0; i < clientPid.size(); ++i) {
        clientTemp.push_back(0);
        clientErrors.push_back(0);
        clientCalcs.push_back(0);
        clientFaulty.push_back(false);
    }

    time_t startTime = time(0);

    int changeCount;
    float nextReport = 10.0f;
    bool childReport = false;
    while (
        (changeCount = select(maxHandle + 1, &waitHandles, NULL, NULL, NULL))) {
        size_t clientsAlive = 0;
        time_t currentTime = time(0);
        for (size_t i = 0; i < clientFd.size(); ++i) {
            if (FD_ISSET(clientFd.at(i), &waitHandles)) {
                // First, reading processed
                int processed, errors;
                int res = read(clientFd.at(i), &processed, sizeof(int));
                if (res < sizeof(int)) {
                    fprintf(stderr, "read[%zu] error %d", i, res);
                    processed = -1;
                }
                // Then errors
                read(clientFd.at(i), &errors, sizeof(int));

                clientErrors.at(i) += errors;
                if (processed == -1)
                    clientCalcs.at(i) = -1;
                else {
                    clientCalcs.at(i) += processed;
                }

                if (errors)
                    clientFaulty.at(i) = true;

                childReport = true;
            }
            if (clientCalcs.at(i) >= 0)
                clientsAlive++;
        }

        // Updating status
        float elapsed =
            fminf((float)(currentTime - startTime) / (float)runLength * 100.0f,
                  100.0f);
        if (elapsed >= nextReport || (childReport && nextReport < 100.0f)) {
            childReport = false;
            printf("\r%.1f%%  ", elapsed);
            printf("proc'd: ");
            if (specificCudaDevice >= 0) {
                // Single GPU mode: show directly
                for (size_t i = 0; i < clientCalcs.size(); ++i) {
                    printf("%d (%.0f Gflop/s)", clientCalcs.at(i),
                           (double)clientCalcs.at(i) * (double)OPS_PER_MUL /
                               fmaxf((double)(currentTime - startTime), 0.01) /
                               1e9);
                    if (i != clientCalcs.size() - 1)
                        printf(" - ");
                }
            } else {
                // Multi-GPU mode: show in nvidia-smi order
                for (size_t nvsmi_idx = 0;
                     nvsmi_idx < nvidiaSmiToCudaMap.size(); ++nvsmi_idx) {
                    int cuda_idx = nvidiaSmiToCudaMap[nvsmi_idx];
                    if (cuda_idx < clientCalcs.size()) {
                        printf(
                            "%d (%.0f Gflop/s)", clientCalcs.at(cuda_idx),
                            (double)clientCalcs.at(cuda_idx) *
                                (double)OPS_PER_MUL /
                                fmaxf((double)(currentTime - startTime), 0.01) /
                                1e9);
                    } else {
                        printf("-- (-- Gflop/s)");
                    }
                    if (nvsmi_idx != nvidiaSmiToCudaMap.size() - 1)
                        printf(" - ");
                }
            }
            printf("  errors: ");
            if (specificCudaDevice >= 0) {
                // Single GPU mode: show directly
                for (size_t i = 0; i < clientErrors.size(); ++i) {
                    printf("%d", clientErrors.at(i));
                    if (i != clientErrors.size() - 1)
                        printf(" - ");
                }
            } else {
                // Multi-GPU mode: show in nvidia-smi order
                for (size_t nvsmi_idx = 0;
                     nvsmi_idx < nvidiaSmiToCudaMap.size(); ++nvsmi_idx) {
                    int cuda_idx = nvidiaSmiToCudaMap[nvsmi_idx];
                    if (cuda_idx < clientErrors.size()) {
                        printf("%d", clientErrors.at(cuda_idx));
                    } else {
                        printf("--");
                    }
                    if (nvsmi_idx != nvidiaSmiToCudaMap.size() - 1)
                        printf(" - ");
                }
            }
            printf("  temps: ");
            if (specificCudaDevice >= 0) {
                // Single GPU mode: show directly
                for (size_t i = 0; i < clientTemp.size(); ++i) {
                    // FIX #1: Corrected printing logic
                    if (clientTemp.at(i) != 0) {
                        printf("%d C", clientTemp.at(i));
                    } else {
                        printf("--");
                    }
                    if (i != clientTemp.size() - 1)
                        printf(" - ");
                }
            } else {
                // Multi-GPU mode: show in nvidia-smi order
                for (size_t nvsmi_idx = 0;
                     nvsmi_idx < nvidiaSmiToCudaMap.size(); ++nvsmi_idx) {
                    int cuda_idx = nvidiaSmiToCudaMap[nvsmi_idx];
                    if (cuda_idx < clientTemp.size()) {
                        // FIX #2: Corrected printing logic here as well
                        if (clientTemp.at(cuda_idx) != 0) {
                            printf("%d C", clientTemp.at(cuda_idx));
                        } else {
                            printf("--");
                        }
                    } else {
                        printf("--");
                    }
                    if (nvsmi_idx != nvidiaSmiToCudaMap.size() - 1)
                        printf(" - ");
                }
            }
            printf("\r");
            fflush(stdout);

            for (size_t i = 0; i < clientErrors.size(); ++i)
                if (clientErrors.at(i))
                    clientFaulty.at(i) = true;

            if (nextReport < elapsed) {
                nextReport = elapsed + 10.0f;
                printf("\n\tSummary at:   ");
                fflush(stdout);
                system("date");
                fflush(stdout);
                printf("\n");
                for (size_t i = 0; i < clientErrors.size(); ++i)
                    clientErrors.at(i) = 0;
            }
        }

        if (currentTime - startTime > runLength)
            break;
        if (!clientsAlive) {
            fprintf(stderr, "\n\nNo clients are alive!  Aborting\n");
            exit(ENOMEDIUM);
        }

        if (FD_ISSET(tempHandle, &waitHandles))
            updateTempsWithMapping(tempHandle, &clientTemp, nvidiaSmiToCudaMap,
                                   specificCudaDevice);

        FD_ZERO(&waitHandles);
        FD_SET(tempHandle, &waitHandles);
        for (size_t i = 0; i < clientFd.size(); ++i) {
            if (clientCalcs.at(i) >= 0)
                FD_SET(clientFd.at(i), &waitHandles);
        }
    }

    printf("\nKilling processes with SIGTERM (soft kill)\n");
    fflush(stdout);
    for (size_t i = 0; i < clientPid.size(); ++i)
        kill(clientPid.at(i), SIGTERM);

    kill(tempPid, SIGTERM);

    std::this_thread::sleep_for(sigterm_timeout_threshold_secs);

    std::vector<int> killed_processes;
    for (size_t i = 0; i < clientPid.size(); ++i) {
        int status;
        pid_t return_pid = waitpid(clientPid.at(i), &status, WNOHANG);
        if (return_pid == clientPid.at(i)) {
            killed_processes.push_back(return_pid);
        }
    }
    int status;
    pid_t return_pid = waitpid(tempPid, &status, WNOHANG);
    if (return_pid == tempPid) {
        killed_processes.push_back(return_pid);
    }

    if (killed_processes.size() != clientPid.size() + 1) {
        printf("\nKilling processes with SIGKILL (force kill)\n");

        for (size_t i = 0; i < clientPid.size(); ++i) {
            if (std::find(killed_processes.begin(), killed_processes.end(),
                          clientPid.at(i)) == killed_processes.end())
                kill(clientPid.at(i), SIGKILL);
        }

        if (std::find(killed_processes.begin(), killed_processes.end(),
                      tempPid) == killed_processes.end())
            kill(tempPid, SIGKILL);
    }

    close(tempHandle);

    while (wait(NULL) != -1)
        ;
    printf("done\n");

    printf("\nTested %d GPUs:\n", (int)clientPid.size());
    if (specificCudaDevice >= 0) {
        int nvsmi_index = 0;
        for (size_t i = 0; i < nvidiaSmiToCudaMap.size(); ++i) {
            if (nvidiaSmiToCudaMap[i] == specificCudaDevice) {
                nvsmi_index = i;
                break;
            }
        }
        printf("\tGPU %d: %s\n", nvsmi_index,
               clientFaulty.at(0) ? "FAULTY" : "OK");
    } else {
        for (size_t nvsmi_idx = 0; nvsmi_idx < nvidiaSmiToCudaMap.size();
             ++nvsmi_idx) {
            int cuda_idx = nvidiaSmiToCudaMap[nvsmi_idx];
            if (cuda_idx < clientFaulty.size()) {
                printf("\tGPU %zu: %s\n", nvsmi_idx,
                       clientFaulty.at(cuda_idx) ? "FAULTY" : "OK");
            }
        }
    }
}

// BF16 CHANGE: Add useBf16 flag to launch function
template <class T>
void launch(int runLength, bool useDoubles, bool useTensorCores, bool useBf16,
            ssize_t useBytes, int device_id, const char * kernelFile,
            std::chrono::seconds sigterm_timeout_threshold_secs) {
#if IS_JETSON
    std::ifstream f_model("/proc/device-tree/model");
    std::stringstream ss_model;
    ss_model << f_model.rdbuf();
    printf("%s\n", ss_model.str().c_str());
#else
    system("nvidia-smi -L");
#endif

    T *A = (T *)malloc(sizeof(T) * SIZE * SIZE);
    T *B = (T *)malloc(sizeof(T) * SIZE * SIZE);
    srand(10);
    for (size_t i = 0; i < SIZE * SIZE; ++i) {
        // This cast sequence works for float, double, and __nv_bfloat16
        A[i] = (T)(float)((double)(rand() % 1000000) / 100000.0);
        B[i] = (T)(float)((double)(rand() % 1000000) / 100000.0);
    }

    int mainPipe[2];
    pipe(mainPipe);
    int readMain = mainPipe[0];
    std::vector<int> clientPipes;
    std::vector<pid_t> clientPids;
    clientPipes.push_back(readMain);
    
    std::vector<int> nvidiaSmiToCudaMap;

    if (device_id > -1) {
        pid_t myPid = fork();
        if (!myPid) {
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            initCuda();
            int devCount = 1;
            write(writeFd, &devCount, sizeof(int));
            
            int totalDevices = 0;
            cuDeviceGetCount(&totalDevices);
            std::vector<int> mapping = createNvidiaSmiToCudaMapping(totalDevices);
            
            int cuda_device_id = device_id;
            if (device_id < mapping.size()) {
                cuda_device_id = mapping[device_id];
            } else {
                fprintf(stderr, "GPU %d not found (nvidia-smi index)\n", device_id);
                exit(EINVAL);
            }
            
            size_t mapSize = mapping.size();
            write(writeFd, &mapSize, sizeof(size_t));
            write(writeFd, mapping.data(), mapSize * sizeof(int));
            
            // BF16 CHANGE: Pass useBf16 flag to startBurn
            startBurn<T>(cuda_device_id, writeFd, A, B, useDoubles, useTensorCores, useBf16,
                         useBytes, kernelFile, device_id);
            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);
            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));
            
            size_t mapSize;
            read(readMain, &mapSize, sizeof(size_t));
            nvidiaSmiToCudaMap.resize(mapSize);
            read(readMain, nvidiaSmiToCudaMap.data(), mapSize * sizeof(int));
            
            int cuda_device_id = device_id < nvidiaSmiToCudaMap.size() ? nvidiaSmiToCudaMap[device_id] : device_id;
            
            listenClients(clientPipes, clientPids, runLength, sigterm_timeout_threshold_secs, nvidiaSmiToCudaMap, cuda_device_id);
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    } else {
        pid_t myPid = fork();
        if (!myPid) {
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            int devCount = initCuda();
            write(writeFd, &devCount, sizeof(int));

            std::vector<int> mapping = createNvidiaSmiToCudaMapping(devCount);
            int nvsmiIndex = 0;
            for (size_t i = 0; i < mapping.size(); i++) {
                if (mapping[i] == 0) {
                    nvsmiIndex = i;
                    break;
                }
            }
            // BF16 CHANGE: Pass useBf16 flag to startBurn
            startBurn<T>(0, writeFd, A, B, useDoubles, useTensorCores, useBf16, useBytes,
                         kernelFile, nvsmiIndex);
            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);
            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));
            if (!devCount) {
                fprintf(stderr, "No CUDA devices\\n");
                exit(ENODEV);
            } else {
                int mappingPipe[2];
                pipe(mappingPipe);

                pid_t mappingPid = fork();
                if (!mappingPid) {
                    close(mappingPipe[0]);
                    initCuda();
                    std::vector<int> mapping =
                        createNvidiaSmiToCudaMapping(devCount);
                    size_t mapSize = mapping.size();
                    write(mappingPipe[1], &mapSize, sizeof(size_t));
                    write(mappingPipe[1], mapping.data(),
                          mapSize * sizeof(int));
                    close(mappingPipe[1]);
                    exit(0);
                }

                close(mappingPipe[1]);
                size_t mapSize;
                read(mappingPipe[0], &mapSize, sizeof(size_t));
                nvidiaSmiToCudaMap.resize(mapSize);
                read(mappingPipe[0], nvidiaSmiToCudaMap.data(),
                     mapSize * sizeof(int));
                close(mappingPipe[0]);
                waitpid(mappingPid, NULL, 0);

                for (int i = 1; i < devCount; ++i) {
                    int slavePipe[2];
                    pipe(slavePipe);

                    int indexPipe[2];
                    pipe(indexPipe);

                    clientPipes.push_back(slavePipe[0]);
                    pid_t slavePid = fork();
                    if (!slavePid) {
                        close(slavePipe[0]);
                        close(indexPipe[1]);

                        int nvsmiIndex;
                        read(indexPipe[0], &nvsmiIndex, sizeof(int));
                        close(indexPipe[0]);

                        initCuda();
                        // BF16 CHANGE: Pass useBf16 flag to startBurn
                        startBurn<T>(i, slavePipe[1], A, B, useDoubles,
                                     useTensorCores, useBf16, useBytes, kernelFile,
                                     nvsmiIndex);
                        close(slavePipe[1]);
                        return;
                    } else {
                        clientPids.push_back(slavePid);
                        close(slavePipe[1]);
                        close(indexPipe[0]);

                        int nvsmiIndex = 0;
                        for (size_t j = 0; j < nvidiaSmiToCudaMap.size(); j++) {
                            if (nvidiaSmiToCudaMap[j] == i) {
                                nvsmiIndex = j;
                                break;
                            }
                        }
                        write(indexPipe[1], &nvsmiIndex, sizeof(int));
                        close(indexPipe[1]);
                    }
                }
                listenClients(clientPipes, clientPids, runLength,
                              sigterm_timeout_threshold_secs,
                              nvidiaSmiToCudaMap);
            }
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    }
    free(A);
    free(B);
}

void showHelp() {
    printf("GPU Burn\n");
    printf("Usage: gpu-burn [OPTIONS] [TIME]\n\n");
    printf("-m X\tUse X MB of memory.\n");
    printf("-m N%%\tUse N%% of the available GPU memory.  Default is %d%%\n",
           (int)(USEMEM * 100));
    printf("-d\tUse doubles\n");
    printf("-bf16\tUse bfloat16\n"); // BF16 CHANGE: Add help text for new option
    printf("-tc\tTry to use Tensor cores\n");
    printf("-l\tLists all GPUs in the system (in nvidia-smi order)\n");
    printf("-i N\tExecute only on GPU N (N is the GPU index from nvidia-smi)\n");
    printf("-c FILE\tUse FILE as compare kernel.  Default is %s\n",
           COMPARE_KERNEL);
    printf("-stts T\tSet timeout threshold to T seconds for using SIGTERM to abort child processes before using SIGKILL.  Default is %d\n",
           SIGTERM_TIMEOUT_THRESHOLD_SECS);
    printf("-h\tShow this help message\n\n");
    printf("Examples:\n");
    printf("  gpu-burn -d 3600 # burns all GPUs with doubles for an hour\n");
    printf(
        "  gpu-burn -m 50%% # burns using 50%% of the available GPU memory\n");
    printf("  gpu-burn -l # list GPUs (in nvidia-smi order)\n");
    printf("  gpu-burn -i 0 # burns only GPU 0 (nvidia-smi index)\n");
}

ssize_t decodeUSEMEM(const char *s) {
    char *s2;
    int64_t r = strtoll(s, &s2, 10);
    if (s == s2)
        return 0;
    if (*s2 == '%')
        return (s2[1] == 0) ? -r : 0;
    return (*s2 == 0) ? r * 1024 * 1024 : 0;
}

int main(int argc, char **argv) {
    int runLength = 10;
    bool useDoubles = false;
    bool useTensorCores = false;
    bool useBf16 = false; // BF16 CHANGE: Add flag for bfloat16 mode
    int thisParam = 0;
    ssize_t useBytes = 0;
    int device_id = -1;
    char *kernelFile = (char *)COMPARE_KERNEL;
    std::chrono::seconds sigterm_timeout_threshold_secs = std::chrono::seconds(SIGTERM_TIMEOUT_THRESHOLD_SECS);

    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < args.size(); ++i) {
        if (argc >= 2 && std::string(argv[i]).find("-h") != std::string::npos) {
            showHelp();
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-l") != std::string::npos) {
            int count = initCuda();
            if (count == 0) {
                throw std::runtime_error("No CUDA capable GPUs found.\n");
            }
            
            std::vector<int> nvidiaSmiToCudaMap = createNvidiaSmiToCudaMapping(count);
            
            for (size_t nvsmi_idx = 0; nvsmi_idx < nvidiaSmiToCudaMap.size(); nvsmi_idx++) {
                int cuda_idx = nvidiaSmiToCudaMap[nvsmi_idx];
                CUdevice device;
                char device_name[255];
                checkError(cuDeviceGet(&device, cuda_idx));
                checkError(cuDeviceGetName(device_name, 255, device));
                std::string uuid = getGpuUuid(cuda_idx);
                printf("GPU %zu: %s (UUID: %s)\n", nvsmi_idx, device_name, uuid.c_str());
            }
            
            thisParam++;
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-d") != std::string::npos) {
            useDoubles = true;
            thisParam++;
        }
        // BF16 CHANGE: Add command line parsing for -bf16
        if (argc >= 2 && std::string(argv[i]).find("-bf16") != std::string::npos) {
            useBf16 = true;
            thisParam++;
        }
        if (argc >= 2 &&
            std::string(argv[i]).find("-tc") != std::string::npos) {
            useTensorCores = true;
            thisParam++;
        }
        if (argc >= 2 && strncmp(argv[i], "-m", 2) == 0) {
            thisParam++;

            if (argv[i][2]) {
                useBytes = decodeUSEMEM(argv[i] + 2);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                useBytes = decodeUSEMEM(argv[i]);
            } else {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
            if (useBytes == 0) {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-i", 2) == 0) {
            thisParam++;

            if (argv[i][2]) {
                device_id = strtol(argv[i] + 2, NULL, 0);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                device_id = strtol(argv[i], NULL, 0);
            } else {
                fprintf(stderr, "Syntax error near -i\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-c", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                kernelFile = argv[i + 1];
                thisParam++;
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-stts", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                sigterm_timeout_threshold_secs = std::chrono::seconds(atoi(argv[i + 1]));
                thisParam++;
            }
        }
    }

    // BF16 CHANGE: Ensure -d and -bf16 are not used together
    if (useDoubles && useBf16) {
        fprintf(stderr, "Error: Cannot use both -d (double) and -bf16 at the same time.\n");
        exit(EINVAL);
    }

    if (argc - thisParam < 2)
        printf("Run length not specified in the command line. ");
    else
        runLength = atoi(argv[1 + thisParam]);
    printf("Using compare file: %s\n", kernelFile);
    printf("Burning for %d seconds.\n", runLength);

    // BF16 CHANGE: Determine which template instantiation to launch
    if (useDoubles)
        launch<double>(runLength, useDoubles, useTensorCores, useBf16, useBytes,
                       device_id, kernelFile, sigterm_timeout_threshold_secs);
    else if (useBf16)
        launch<__nv_bfloat16>(runLength, useDoubles, useTensorCores, useBf16, useBytes,
                              device_id, kernelFile, sigterm_timeout_threshold_secs);
    else
        launch<float>(runLength, useDoubles, useTensorCores, useBf16, useBytes,
                      device_id, kernelFile, sigterm_timeout_threshold_secs);

    return 0;
}
