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
#include <algorithm>

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
void printDataset(Dataset *data);


Dataset* generateFolds(Dataset* data);
void shuffleData(Dataset* data, int* seed);
void getIndexes(int* indices, int k, int excludeIndex, int* seed);
Dataset* getSelectedDataset(Dataset* folds, int* indexes, int index_start, int index_end);
void transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs);


void setupOpenCLOnePlatform(std::vector<cl::Platform> &platforms, std::vector<cl::Device> &devices);

std::string setProgramSource(Dataset* data, Parameters* p, int localSize);
std::string setProgramSource(Dataset* train, Dataset* valid, Parameters* p, int localSize);

void setNDRanges(size_t* globalSize, size_t* localSize, size_t maxLocalSize, size_t numPoints, cl_device_type deviceType);

void setCompileFlags(std::string* compileFlags, size_t localSize, size_t numPoints, int validation);
void setCompileFlags(std::string* compileFlags,size_t localSize, size_t numPoints, size_t numPoints_valid);


void printOpenclDeviceInfo(std::vector<cl::Platform> platforms, std::vector<cl::Device> devices);
const char *getErrorString(cl_int error);
void checkError(cl_int result);


std::string ToString( float t );
bool IsPowerOf2( int n );
unsigned NextPowerOf2( unsigned n );
bool stopCriteria(unsigned int it);

#endif //PCGP_UTILS_H
