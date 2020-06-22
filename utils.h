//
// Created by bruno on 04/02/2020.
//

#ifndef PCGP_UTILS_H
#define PCGP_UTILS_H

/** C headers */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <random>

/** CPP headers */
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <limits>
#include <climits>
#include <ctime>
#include <sstream>
#include <fstream>

#include "constants.h"


#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>


#define GPU_PLATFORM 0
#define CPU_PLATFORM 1

#define GPU_DEVICE 0
#define CPU_DEVICE 1



unsigned int randomFunction(Parameters *p, int *seed);

unsigned int randomInput(Parameters *p, unsigned int index, int *seed);

float randomConnectionWeight(Parameters *p, int *seed);

int randomInterval(int inf_bound, int sup_bound, int *seed);
float randomProb(int *seed);

unsigned int randomOutputIndex(int* seed);
unsigned int getFunctionInputs(unsigned int function);

void readDataset(Parameters* params, float*** dataset, float*** outputs, char* filename);
void readDataset(Parameters* params, Dataset* fulldata, char* filename);
void printDataset(Parameters *params, float **dataset, float **outputs);


bool IsPowerOf2( int n );
unsigned NextPowerOf2( unsigned n );
bool stopCriteria(unsigned int it);


Dataset* generateFolds(float** dataset, float** outputs, Parameters* p);
Dataset* generateFolds(Dataset* data);

void transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs);

const char *getErrorString(cl_int error);
void setupOpenCLOnePlatform(std::vector<cl::Platform> &platforms, std::vector<cl::Device> &devices);
std::string ToString( float t );
std::string setProgramSource(Dataset* data, Parameters* p, int localSize);
void setNDRanges(size_t* globalSize, size_t* localSize, std::string* compileFlags, size_t maxLocalSize, size_t numPoints, cl_device_type deviceType);
#endif //PCGP_UTILS_H
