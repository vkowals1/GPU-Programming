//
// Module 10
//
// Course: Introduction to GPU Programming - EN.605.617.81.SP23
// Student: Vincent Kowalski
// Date: April 8, 2023
//
// OpenCL
//
//
//
//
// File:       hello.c
//
// Purpose:    A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the following arithmetic functions for a buffer of
//             floating point values:
//             - add
//             - subtract
//             - multiply
//             - divide
//             - power (i.e., exponentiation)
//
//             This application is based on the Hello World application provided with the
//             OpenCL Programming Guide textbook by Munshi, et. al. The original documentation
//             is provided below
//
// Approach:   This application was developed in C/C++ on a MacBook Air with an Apple M1 Chip
//             with 7 GPUs. As it was targeted for this hardware platform some modifications to
//             the original source code were made. Each of the arithmetic functions are included
//             in the source code as a string rather than reading them in from a file. Some
//             variable names were changed in order to better accomodate the fact that the app
//             implements 5 different kernel functions.
//
//
//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//
//
// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

//
//
//
//


////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

// Simple add computer kernel function
//
const char *KernelSource1 = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < int(count))                                                  \n" \
"       output[i] = input[i] + input[i];                                \n" \
"}                                                                      \n" \
"\n";

// Simple subtract computer kernel function
//
const char *KernelSource2 = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < int(count))                                                  \n" \
"       output[i] = input[i] - input[i];                                \n" \
"}                                                                      \n" \
"\n";

// Simple multiply computer kernel function
//
const char *KernelSource3 = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < int(count))                                                  \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

// Simple divide computer kernel function
//
const char *KernelSource4 = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < int(count))                                                  \n" \
"       output[i] = input[i] / input[i];                                \n" \
"}                                                                      \n" \
"\n";

// Simple exponentiation computer kernel function
//
const char *KernelSource5 = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count,                                           \n" \
"   int exponent)                                                       \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < int(count))                                                  \n" \
"      //output[i] = input[i] * input[i] * input[i];                    \n" \
"      output[i] = input[i];                                            \n" \
"      for (int j = 1; j < exponent; j++)                               \n" \
"         output[i] = output[i] * input[i];                             \n" \
"}                                                                      \n" \
"\n";


////////////////////////////////////////////////////////////////////////////////


// Main Program
int main(int argc, char** argv)
{
    // Declarations of Variables
    int err;                                             // error code returned from api calls
      
    float data[DATA_SIZE];                               // original data set given to device
    float results1[DATA_SIZE], results2[DATA_SIZE],
          results3[DATA_SIZE], results4[DATA_SIZE],
          results5[DATA_SIZE];          // results returned from device
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    cl_context context1, context2, context3, context4, context5;                // compute context
    cl_command_queue commands1, commands2, commands3, commands4, commands5;     // compute command queue
    cl_program program1, program2, program3, program4, program5;                // compute program
    cl_kernel kernel1, kernel2, kernel3, kernel4, kernel5;                      // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    // Fill our data set with random float values
    //
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;
    
    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    context1 = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context1)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands1 = clCreateCommandQueue(context1, device_id, 0, &err);
    if (!commands1)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program1 = clCreateProgramWithSource(context1, 1, (const char **) & KernelSource1, NULL, &err);
    if (!program1)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program1, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program1, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel1 = clCreateKernel(program1, "square", &err);
    if (!kernel1 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context1,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context1, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands1, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel1, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel1, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands1, kernel1, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands1);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands1, output, CL_TRUE, 0, sizeof(float) * count, results1, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results1[i] == data[i] + data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values for the add function.\n", correct, count);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program1);
    clReleaseKernel(kernel1);
    clReleaseCommandQueue(commands1);
    clReleaseContext(context1);
    
    
    // Create a second compute context
    //
    context2 = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context2)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands2 = clCreateCommandQueue(context2, device_id, 0, &err);
    if (!commands2)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program2 = clCreateProgramWithSource(context2, 1, (const char **) & KernelSource2, NULL, &err);
    if (!program2)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program2, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program2, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel2 = clCreateKernel(program2, "square", &err);
    if (!kernel2 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context2,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context2, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands2, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel2, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel2, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands2, kernel2, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands2);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands2, output, CL_TRUE, 0, sizeof(float) * count, results2, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results2[i] == data[i] - data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values for the subtract function.\n", correct, count);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program2);
    clReleaseKernel(kernel2);
    clReleaseCommandQueue(commands2);
    clReleaseContext(context2);
    
    // Create a third compute context
    //
    context3 = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context3)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands3 = clCreateCommandQueue(context3, device_id, 0, &err);
    if (!commands3)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program3 = clCreateProgramWithSource(context3, 1, (const char **) & KernelSource3, NULL, &err);
    if (!program3)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program3, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program3, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel3 = clCreateKernel(program3, "square", &err);
    if (!kernel3 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context3,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context3, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands3, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel3, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel3, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands3, kernel3, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands3);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands3, output, CL_TRUE, 0, sizeof(float) * count, results3, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results3[i] == data[i] * data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values for the multiply function.\n", correct, count);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program3);
    clReleaseKernel(kernel3);
    clReleaseCommandQueue(commands3);
    clReleaseContext(context3);
    
    // Create a fourth compute context
    //
    context4 = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context4)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands4 = clCreateCommandQueue(context4, device_id, 0, &err);
    if (!commands4)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program4 = clCreateProgramWithSource(context4, 1, (const char **) & KernelSource4, NULL, &err);
    if (!program4)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program4, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program4, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel4 = clCreateKernel(program4, "square", &err);
    if (!kernel4 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context4,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context4, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands4, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel4, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel4, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel4, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel4, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands4, kernel4, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands4);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands4, output, CL_TRUE, 0, sizeof(float) * count, results4, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results4[i] == data[i] / data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values for the divide function.\n", correct, count);
    
    // Create a fifth compute context
    //
    context5 = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context5)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands5 = clCreateCommandQueue(context5, device_id, 0, &err);
    if (!commands5)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program5 = clCreateProgramWithSource(context5, 1, (const char **) & KernelSource5, NULL, &err);
    if (!program5)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program5, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program4, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel5 = clCreateKernel(program5, "square", &err);
    if (!kernel5 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context5,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context5, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands5, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    int exponent = 3;
    err  = clSetKernelArg(kernel5, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel5, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel5, 2, sizeof(unsigned int), &count);
    err |= clSetKernelArg(kernel5, 3, sizeof( int), &exponent);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel5, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands5, kernel5, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands4);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands5, output, CL_TRUE, 0, sizeof(float) * count, results5, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    //int exponent = 3;
    for(i = 0; i < count; i++)
    {
        if(results5[i] == data[i] * data[i] * data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values for the power function.\n", correct, count);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program5);
    clReleaseKernel(kernel5);
    clReleaseCommandQueue(commands5);
    clReleaseContext(context5);

    return 0;
}

