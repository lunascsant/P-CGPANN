//
// Created by bruno on 25/06/2020.
//

#ifndef P_CGPDE_OCLCONFIG_H
#define P_CGPDE_OCLCONFIG_H

#include "constants.h"
#include "utils.h"

class OCLConfig {
    public:
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;

        cl::Context context;
        cl::CommandQueue cmdQueue;
        cl::Program program;
        cl::Kernel kernel;

        ///Buffers
        cl::Buffer bufferSeeds;

        cl::Buffer bufferDatasetTrain;
        cl::Buffer bufferOutputsTrain;

        cl::Buffer bufferDatasetValid;
        cl::Buffer bufferOutputsValid;

        cl::Buffer bufferDatasetTest;
        cl::Buffer bufferOutputsTest;

        cl::Buffer bufferFunctions;

        cl::Buffer bufferBest;
        cl::Buffer bufferNewBest;

        size_t numPointsTrain;
        size_t numPointsValid;
        size_t numPointsTest;

        size_t localSize_aval;
        size_t globalSize_aval;

        std::string compileFlags;
        cl_command_queue_properties commandQueueProperties;






};


#endif //P_CGPDE_OCLCONFIG_H
