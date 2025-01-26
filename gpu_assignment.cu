#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <sys/time.h>

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float pi = 3.14159265;
long int MemoryAllocatedCPU = 0L;

__global__ void Hist(float *real_rasc, float *real_decl, float *rand_rasc, float *rand_decl, unsigned long long int *histogramDR, unsigned long long int *histogramDD, unsigned long long int *histogramRR) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    const float PI = acosf(-1.0f);

    // Ensure that we do not go out of bounds
    if (index < 100000) {
        float real_rasc_val = real_rasc[index];
        float real_decl_val = real_decl[index];
        float tempDR, tempDD, tempRR;
        float angleDR, angleDD, angleRR;

        for (int j = 0; j < 100000; j++) {
            // Compute the cosine of the angle between two sets of coordinates
            tempDR = sinf(real_decl_val) * sinf(rand_decl[j]) + cosf(real_decl_val) * cosf(rand_decl[j]) * cosf(real_rasc_val - rand_rasc[j]);
            angleDR = tempDR >= 1 ? 0 : 180.0f / PI * acosf(tempDR);
            tempDD = sinf(real_decl_val) * sinf(real_decl[j]) + cosf(real_decl_val) * cosf(real_decl[j]) * cosf(real_rasc_val - real_rasc[j]);
            angleDD = tempDD >= 1 ? 0 : 180.0f / PI * acosf(tempDD);
            tempRR = sinf(rand_decl[index]) * sinf(rand_decl[j]) + cosf(rand_decl[index]) * cosf(rand_decl[j]) * cosf(rand_rasc[index] - rand_rasc[j]);
            angleRR = tempRR >= 1 ? 0 : 180.0f / PI * acosf(tempRR);

            // Add the calculated angle to corresponding histogram bin
            atomicAdd(&histogramDR[(int)(angleDR * 4.0f)], 1LL);
            atomicAdd(&histogramDD[(int)(angleDD * 4.0f)], 1LL);
            atomicAdd(&histogramRR[(int)(angleRR * 4.0f)], 1LL);
        }
    }
}

int get_data(int argc, char *argv[]) {
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f / 60.0f / 180.0f * pi;
    int Number_of_Galaxies;

    if (argc != 4) {
        printf("   Usage: galaxy real_data flat_data output_file\n   All processes will be killed\n");
        return(1);
    }
    if (argc == 4) {
        printf("   Running galaxy_cuda %s %s %s\n", argv[1], argv[2], argv[3]);

        real_data_file = fopen(argv[1], "r");
        if (real_data_file == NULL) {
            printf("   ERROR: Cannot open real data file %s\n", argv[1]);
            return(1);
        } else {
            fscanf(real_data_file, "%d", &Number_of_Galaxies);
            if (Number_of_Galaxies != 100000) {
                printf("Cannot read file %s correctly, first item not 100000\n", argv[1]);
                fclose(real_data_file);
                return(1);
            }
            for (int i = 0; i < 100000; ++i) {
                float rasc, decl;
                if (fscanf(real_data_file, "%f %f", &rasc, &decl) != 2) {
                    printf("   ERROR: Cannot read line %d in real data file %s\n", i + 1, argv[1]);
                    fclose(real_data_file);
                    return(1);
                }
                real_rasc[i] = rasc * arcmin2rad;
                real_decl[i] = decl * arcmin2rad;
            }
            fclose(real_data_file);
            printf("   Successfully read 100000 lines from %s\n", argv[1]);
        }

        rand_data_file = fopen(argv[2], "r");
        if (rand_data_file == NULL) {
            printf("   ERROR: Cannot open random data file %s\n", argv[2]);
            return(1);
        } else {
            fscanf(rand_data_file, "%d", &Number_of_Galaxies);
            if (Number_of_Galaxies != 100000) {
                printf("Cannot read file %s correctly, first item not 100000\n", argv[2]);
                fclose(rand_data_file);
                return(1);
            }
            for (int i = 0; i < 100000; ++i) {
                float rasc, decl;
                if (fscanf(rand_data_file, "%f %f", &rasc, &decl) != 2) {
                    printf("   ERROR: Cannot read line %d in random data file %s\n", i + 1, argv[2]);
                    fclose(rand_data_file);
                    return(1);
                }
                rand_rasc[i] = rasc * arcmin2rad;
                rand_decl[i] = decl * arcmin2rad;
            }
            fclose(rand_data_file);
            printf("   Successfully read 100000 lines from %s\n", argv[2]);
        }
        
        out_file = fopen(argv[3], "w");
        if (out_file == NULL) {
            printf("   ERROR: Cannot open output file %s\n", argv[3]);
            return(1);
        } else fclose(out_file);
    }

    return(0);
}

int getDevice(void) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128) return (-1);
    
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name, device);
        printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory            =        %.2f GB\n", deviceProp.totalGlobalMem / 1000000000.0f);
        printf("         multiProcessorCount          =         %d\n", deviceProp.multiProcessorCount);
        printf("         memoryClockRate              =        %.0f MHz\n", deviceProp.memoryClockRate / 1000.0f);
        printf("         memoryBusWidth               =         %d bits\n", deviceProp.memoryBusWidth);
        printf("         warpSize                     =         %d\n", deviceProp.warpSize);
    }

    cudaSetDevice(0);
    cudaGetDevice(&device);
    if (device != 0) {
        printf("   Unable to set device 0, using %d instead\n", device);
    } else {
        printf("   Using CUDA device %d\n\n", device);
    }
    return 0;
}

int main(int argc, char* argv[]) {
    int get_data(int argc, char *argv[]);

    cudaEvent_t start, stop;
    float elapsedTime;

    // Record start time using CUDA event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (getDevice() != 0) return (-1);

    real_rasc = (float *)calloc(100000L, sizeof(float));
    real_decl = (float *)calloc(100000L, sizeof(float));
    rand_rasc = (float *)calloc(100000L, sizeof(float));
    rand_decl = (float *)calloc(100000L, sizeof(float));
    MemoryAllocatedCPU += 10L * 100000L * sizeof(float);
    
    if (get_data(argc, argv) != 0) {
        printf("   Program stopped.\n");
        return (0);
    }
    
    printf("   Input data read, now calculating histograms\n");

    long int histogram_DD[360] = {0L};
    long int histogram_DR[360] = {0L};
    long int histogram_RR[360] = {0L};
    MemoryAllocatedCPU += 3L * 360L * sizeof(long int);

    size_t data_size = 100000 * sizeof(float);
    size_t histo_size = 360 * sizeof(unsigned long long int);
    
    float *real_rasc_gpu; 
    cudaMalloc(&real_rasc_gpu, data_size);
    float *real_decl_gpu; 
    cudaMalloc(&real_decl_gpu, data_size);
    float *rand_rasc_gpu; 
    cudaMalloc(&rand_rasc_gpu, data_size);
    float *rand_decl_gpu; 
    cudaMalloc(&rand_decl_gpu, data_size);
    unsigned long long int *histogramDR_gpu; 
    cudaMalloc(&histogramDR_gpu, histo_size);
    unsigned long long int *histogramDD_gpu; 
    cudaMalloc(&histogramDD_gpu, histo_size);
    unsigned long long int *histogramRR_gpu; 
    cudaMalloc(&histogramRR_gpu, histo_size);
    
    cudaMemset(histogramDR_gpu, 0, histo_size);
    cudaMemset(histogramRR_gpu, 0, histo_size);
    cudaMemset(histogramDD_gpu, 0, histo_size);
    
    cudaMemcpy(real_rasc_gpu, real_rasc, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(real_decl_gpu, real_decl, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(rand_rasc_gpu, rand_rasc, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(rand_decl_gpu, rand_decl, data_size, cudaMemcpyHostToDevice);
    
    int threadsInBlock = 256;
    int blocksInGrid = (100000 + threadsInBlock - 1) / threadsInBlock;

    // Launch the CUDA kernel
    Hist<<<blocksInGrid, threadsInBlock>>>(real_rasc_gpu, real_decl_gpu, rand_rasc_gpu, rand_decl_gpu, histogramDR_gpu, histogramDD_gpu, histogramRR_gpu);

    // Check for any errors in the kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Ensure GPU finishes before measuring end time
    cudaDeviceSynchronize();

    // Transfer the computed histograms back to the host
    cudaMemcpy(histogram_DD, histogramDD_gpu, histo_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogram_DR, histogramDR_gpu, histo_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogram_RR, histogramRR_gpu, histo_size, cudaMemcpyDeviceToHost);

    // Verify the values in the histogram to make sure computations went correctly
    long int histsum = 0L;
    int correct_value = 1;
    
    for (int i = 0; i < 360; ++i) histsum += histogram_DD[i];
    printf("   Histogram DD : sum = %ld\n", histsum);
    if (histsum != 10000000000L) correct_value = 0;

    histsum = 0L;
    for (int i = 0; i < 360; ++i) histsum += histogram_DR[i];
    printf("   Histogram DR : sum = %ld\n", histsum);
    if (histsum != 10000000000L) correct_value = 0;

    histsum = 0L;
    for (int i = 0; i < 360; ++i) histsum += histogram_RR[i];
    printf("   Histogram RR : sum = %ld\n", histsum);
    if (histsum != 10000000000L) correct_value = 0;

    if (correct_value != 1) {
        printf("   Histogram sums should be 10000000000. Ending program prematurely\n");
        return (0);
    }

    // Print some omega values and results
    printf("   Some Omega values for the histograms are given below (For all, please check result file):\n");
    float omega[360];
    for (int i = 0; i < 360; ++i)
        if (histogram_RR[i] != 0L) {
            omega[i] = (histogram_DD[i] - 2L * histogram_DR[i] + histogram_RR[i]) / ((float)(histogram_RR[i]));
            if (i < 5) printf("      Angle %.2f degree => %.2f degree: %.3f\n", i * 0.25f, (i + 1) * 0.25f, omega[i]);
        }

    // Open output file
    FILE *out_file = fopen(argv[3], "w");
    if (out_file == NULL) {
        printf("   ERROR: Cannot open output file %s\n", argv[3]);
    } else {
        // Print the headers for the table
        fprintf(out_file, "Bin\tDR\tDD\tRR\tOmega\n");
        
        // Print the histogram data and Omega values
        for (int i = 0; i < 360; ++i) {
            if (histogram_RR[i] != 0L) { // If RR is non-zero, write data to file
                omega[i] = (histogram_DD[i] - 2L * histogram_DR[i] + histogram_RR[i]) / ((float)(histogram_RR[i]));
                
                // Write the bin (angle), DR, DD, RR, and Omega values to the file
                fprintf(out_file, "%d\t%ld\t%ld\t%ld\t%.6f\n", 
                        i,                             // Bin index
                        histogram_DR[i],                // DR count
                        histogram_DD[i],                // DD count
                        histogram_RR[i],                // RR count
                        omega[i]);                      // Omega value
            }
        }
        fclose(out_file);
    }

    // Record end time and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("   Execution time is %.3f milliseconds.\n", elapsedTime);

    return 0;
}
