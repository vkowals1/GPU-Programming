//
//
//  Convolution.cpp
//  convolution
//
//  Created by VINCENT KOWALSKI on 4/11/23.

//
// Module 11
//
// Course: Introduction to GPU Programming - EN.605.617.81.SP23
// Student: Vincent Kowalski
// Date: April 12, 2023
//
// OpenCL - Convolution Application
//
//
//
//
// File:       convolution.cpp
//
// Purpose:    A Convolution calculation using OpenCL.
//
//             This application is based on the Convolution.cpp application provided in Chapter 3
//             of the OpenCL Programming Guide textbook by Munshi, et. al.
//
// Approach:   This application was developed in C/C++ on a MacBook Air with an Apple M1 Chip
//             with 7 GPUs. As it was targeted for this hardware platform some modifications to
//             the original source code were made. The convolve function is included
//             in the source code as a string rather than reading them in from a file. Some
//             variable names were changed in order to better accomodate this design.
//
//             The original logic provided in the textbook is included to form a baseline for
//             the extension that follows. Variables and objects used in the origal textbook
//             application have a '1' embedded in them, e.g., InputSignal1Width, OutputSignal1Height, etc.
//             And variables and objects that are used to implement the modification have integers
//             greater than 1 embedded in their names, e.g., InputSignal2Width, OutputSignal2Height, etc.
//
//
//
// Book:      OpenCL(R) Programming Guide
// Chapter:   Chapter 3 Platforms, Contexts and Devices
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com



// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int InputSignal1Width  = 8;
const unsigned int InputSignal1Height = 8;

cl_uint InputSignal1[InputSignal1Height][InputSignal1Width] =
{
    {3, 1, 1, 4, 8, 2, 1, 3},
    {4, 2, 1, 1, 2, 1, 2, 3},
    {4, 4, 4, 4, 3, 2, 2, 2},
    {9, 8, 3, 8, 9, 0, 0, 0},
    {9, 3, 3, 9, 0, 0, 0, 0},
    {0, 9, 0, 8, 0, 0, 0, 0},
    {3, 0, 8, 8, 9, 4, 4, 4},
    {5, 9, 8, 1, 8, 1, 1, 1}
};

const unsigned int InputSignal2Width  = 49;
const unsigned int InputSignal2Height = 49;

cl_uint InputSignal2[InputSignal2Height][InputSignal2Width] =
{
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 7}
};

const unsigned int OutputSignal1Width  = 6;
const unsigned int OutputSignal1Height = 6;

const unsigned int OutputSignal2Width  = 6;
const unsigned int OutputSignal2Height = 6;

cl_uint OutputSignal1[OutputSignal1Height][OutputSignal1Width];
cl_uint OutputSignal2[OutputSignal2Height][OutputSignal2Width];


const unsigned int mask1Width  = 3;
const unsigned int mask1Height = 3;

const unsigned int mask2Width  = 7;
const unsigned int mask2Height = 7;


cl_uint mask1[mask1Height][mask1Width] =
{
    {1, 1, 1}, {1, 0, 1}, {1, 1, 1},
};

cl_uint mask2[mask2Height][mask2Width] =
{
    {1, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 0, 1, 0, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 0, 1, 0, 1, 0, 1},
    {1, 0, 0, 0, 0, 0, 1}
};


// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
    const char * errInfo,
    const void * private_info,
    size_t cb,
    void * user_data)
{
    std::cout << "Error occured during context use: " << errInfo << std::endl;
    // should really perform any clearup and so on at this point
    // but for simplicitly just exit.
    exit(1);
}

//    main() for Convoloution example
//
int main(int argc, char** argv)
{
    std::cout << "Before Declaration in main() \n" << std::endl;
    
    cl_int errNum, errNum1, errNum2;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context1 = NULL;
    cl_context context2 = NULL;

    cl_command_queue queue1, queue2;
    cl_program program1, program2;
    cl_kernel kernel1, kernel2;
    cl_mem InputSignal1Buffer, InputSignal2Buffer;
    cl_mem OutputSignal1Buffer, OutputSignal2Buffer;
    cl_mem mask1Buffer, mask2Buffer;

    // First, select an OpenCL platform to run on.
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");
 
    platformIDs = (cl_platform_id *)alloca(
               sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
       "clGetPlatformIDs");

    // Iterate through the list of platforms until we find one that supports
    // a CPU device, otherwise fail with an error.
    deviceIDs = NULL;
    cl_uint i;
    for (i = 0; i < numPlatforms; i++)
    {
        errNum = clGetDeviceIDs(
            platformIDs[i],
            CL_DEVICE_TYPE_GPU,
            0,
            NULL,
            &numDevices);
        if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
        {
            checkErr(errNum, "clGetDeviceIDs");
        }
        else if (numDevices > 0)
        {
               deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
            errNum = clGetDeviceIDs(
                platformIDs[i],
                CL_DEVICE_TYPE_GPU,
                numDevices,
                &deviceIDs[0],
                NULL);
            checkErr(errNum, "clGetDeviceIDs");
            break;
       }
    }

    // Check to see if we found at least one CPU device, otherwise return
     if (deviceIDs == NULL) {
         std::cout << "No CPU device found" << std::endl;
         exit(-1);
     }

    // Next, create an OpenCL context on the selected platform.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    
    // create context objects
    context1 = clCreateContext(
        contextProperties,
        numDevices,
        deviceIDs,
        &contextCallback,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateContext");
    
    context2 = clCreateContext(
        contextProperties,
        numDevices,
        deviceIDs,
        &contextCallback,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateContext");


    //std::ifstream srcFile("Convolution.cl");
    //checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

    //std::string srcProg(
    //    std::istreambuf_iterator<char>(srcFile),
    //    (std::istreambuf_iterator<char>()));

   // const char * src = srcProg.c_str();
    
    
    const char *src1 = "\n" \
    "__kernel void convolve(                                                \n" \
    "                       const __global uint *const input,               \n" \
    "                       __constant uint *const mask,                    \n" \
    "                       __global uint *const output,                    \n" \
    "                       const int inputWidth,                           \n" \
    "                       const int maskWidth)                            \n" \
    "{                                                                      \n" \
    "    const int x = get_global_id(0);                                    \n" \
    "    const int y = get_global_id(1);                                    \n" \
    "                                                                       \n" \
    "    uint sum = 0;                                                      \n" \
    "    for (int r = 0; r < maskWidth; r++)                                \n" \
    "    {                                                                  \n" \
    "        const int idxIntmp = (y + r) * inputWidth + x;                 \n" \
    "                                                                       \n" \
    "        for (int c = 0; c < maskWidth; c++)                            \n" \
    "        {                                                              \n" \
    "            sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];   \n" \
    "        }                                                              \n" \
    "    }                                                                  \n" \
    "                                                                       \n" \
    "    output[y * get_global_size(0) + x] = sum;                          \n" \
    "}                                                                      \n" \
    "\n";

    
    
    //size_t length = src.length();

    // Create program from source
    std::cout << "Before create program in main() \n" << std::endl;
    program1 = clCreateProgramWithSource(
        context1,
        1,
        &src1,
        NULL,
        &errNum1);
    checkErr(errNum1, "clCreateProgramWithSource");

    std::cout << "Before build program in main() \n" << std::endl;
    // Build program
    errNum1 = clBuildProgram(
        program1,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        NULL);
    
    std::cout << "Before error check in main() \n" << std::endl;
    if (errNum1 != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program1,
            deviceIDs[0],
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog),
            buildLog,
            NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }

    
    std::cout << "Before kernel creation in main() \n" << std::endl;
    // Create kernel object
    kernel1 = clCreateKernel(
        program1,
        "convolve",
        &errNum);
    checkErr(errNum1, "clCreateKernel");

    std::cout << "Before input buffer creation in main() \n" << std::endl;
    // Now allocate buffers
    InputSignal1Buffer = clCreateBuffer(
        context1,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * InputSignal1Height * InputSignal1Width,
        static_cast<void *>(InputSignal1),
        &errNum);
    checkErr(errNum1, "clCreateBuffer(InputSignal1)");

    mask1Buffer = clCreateBuffer(
        context1,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * mask1Height * mask1Width,
        static_cast<void *>(mask1),
        &errNum1);
    checkErr(errNum, "clCreateBuffer(mask1)");
    
    std::cout << "Before output buffer creation in main() \n" << std::endl;
    OutputSignal1Buffer = clCreateBuffer(
        context1,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * OutputSignal1Height * OutputSignal1Width,
        NULL,
        &errNum1);
    checkErr(errNum, "clCreateBuffer(OutputSignal1)");

    // Pick the first device and create command queue.
    queue1 = clCreateCommandQueue(
        context1,
        deviceIDs[0],
        0,
        &errNum1);
    checkErr(errNum, "clCreateCommandQueue");

    errNum  = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &InputSignal1Buffer);
    errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &mask1Buffer);
    errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), &OutputSignal1Buffer);
    errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &InputSignal1Width);
    errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_uint), &mask1Width);
    checkErr(errNum1, "clSetKernelArg");

    const size_t globalWorkSize[2] = { OutputSignal1Width, OutputSignal1Height };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
        queue1,
        kernel1,
        2,
        NULL,
        globalWorkSize,
        localWorkSize,
        0,
        NULL,
        NULL);
    checkErr(errNum1, "clEnqueueNDRangeKernel");
    
    errNum = clEnqueueReadBuffer(
        queue1,
        OutputSignal1Buffer,
        CL_TRUE,
        0,
        sizeof(cl_uint) * OutputSignal1Height * OutputSignal1Height,
        OutputSignal1,
        0,
        NULL,
        NULL);
    checkErr(errNum1, "clEnqueueReadBuffer");

    // Output the result buffer
    for (int y = 0; y < OutputSignal1Height; y++)
    {
        for (int x = 0; x < OutputSignal1Width; x++)
        {
            std::cout << OutputSignal1[y][x] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Executed program succesfully." << std::endl;
    
    // Second execution with 49x49 input signal and 7x7 mask
    
    
    const char *src2 = "\n" \
    "__kernel void convolve(                                                \n" \
    "                       const __global uint *const input,               \n" \
    "                       __constant uint *const mask,                    \n" \
    "                       __global uint *const output,                    \n" \
    "                       const int inputWidth,                           \n" \
    "                       const int maskWidth)                            \n" \
    "{                                                                      \n" \
    "    const int x = get_global_id(0);                                    \n" \
    "    const int y = get_global_id(1);                                    \n" \
    "                                                                       \n" \
    "    uint sum = 0;                                                      \n" \
    "    for (int r = 0; r < maskWidth; r++)                                \n" \
    "    {                                                                  \n" \
    "        const int idxIntmp = (y + r) * inputWidth + x;                 \n" \
    "                                                                       \n" \
    "        for (int c = 0; c < maskWidth; c++)                            \n" \
    "        {                                                              \n" \
    "            sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];   \n" \
    "        }                                                              \n" \
    "    }                                                                  \n" \
    "                                                                       \n" \
    "    output[y * get_global_size(0) + x] = sum;                          \n" \
    "}                                                                      \n" \
    "\n";


    //size_t length = src.length();

    // Create program from source
    program2 = clCreateProgramWithSource(
        context2,
        1,
        &src2,
        NULL,
        &errNum2);
    checkErr(errNum2, "clCreateProgramWithSource");

    // Build program
    errNum2 = clBuildProgram(
        program2,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        NULL);
    if (errNum2 != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program2,
            deviceIDs[0],
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog),
            buildLog,
            NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum2, "clBuildProgram");
    }

    // Create kernel object
    kernel2 = clCreateKernel(
        program2,
        "convolve",
        &errNum2);
    checkErr(errNum2, "clCreateKernel");

    // Now allocate buffers
    InputSignal2Buffer = clCreateBuffer(
        context2,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * InputSignal2Height * InputSignal2Width,
        static_cast<void *>(InputSignal2),
        &errNum2);
    checkErr(errNum2, "clCreateBuffer(InputSignal2)");

    mask2Buffer = clCreateBuffer(
        context2,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * mask2Height * mask2Width,
        static_cast<void *>(mask2),
        &errNum2);
    checkErr(errNum2, "clCreateBuffer(mask2)");

    OutputSignal2Buffer = clCreateBuffer(
        context2,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * OutputSignal2Height * OutputSignal2Width,
        NULL,
        &errNum2);
    checkErr(errNum2, "clCreateBuffer(OutputSignal2)");

    // Pick the first device and create command queue.
    queue2 = clCreateCommandQueue(
        context2,
        deviceIDs[0],
        0,
        &errNum2);
    checkErr(errNum2, "clCreateCommandQueue");

    errNum2  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &InputSignal2Buffer);
    errNum2 |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &mask2Buffer);
    errNum2 |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &OutputSignal2Buffer);
    errNum2 |= clSetKernelArg(kernel2, 3, sizeof(cl_uint), &InputSignal2Width);
    errNum2 |= clSetKernelArg(kernel2, 4, sizeof(cl_uint), &mask2Width);
    checkErr(errNum2, "clSetKernelArg");

    //const size_t globalWorkSize[2] = { OutputSignal2Width, OutputSignal2Height };
    //const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum2 = clEnqueueNDRangeKernel(
        queue2,
        kernel2,
        2,
        NULL,
        globalWorkSize,
        localWorkSize,
        0,
        NULL,
        NULL);
    checkErr(errNum2, "clEnqueueNDRangeKernel");

    errNum2 = clEnqueueReadBuffer(
        queue2,
        OutputSignal2Buffer,
        CL_TRUE,
        0,
        sizeof(cl_uint) * OutputSignal2Height * OutputSignal2Height,
        OutputSignal2,
        0,
        NULL,
        NULL);
    checkErr(errNum, "clEnqueueReadBuffer");

    // Output the result buffer
    for (int y = 0; y < OutputSignal2Height; y++)
    {
        for (int x = 0; x < OutputSignal2Width; x++)
        {
            std::cout << OutputSignal2[y][x] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Executed program succesfully." << std::endl;

    return 0;
}


