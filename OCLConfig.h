//
// Created by bruno on 25/06/2020.
//

#ifndef P_CGPDE_OCLCONFIG_H
#define P_CGPDE_OCLCONFIG_H

#include "constants.h"
#include "utils.h"

class OCLConfig {
public:
    OCLConfig();

    cl_uint platformCount;
    cl_platform_id *platforms;

    cl_uint* deviceCount;
    cl_device_id** devices;

    cl_context context;
    cl_command_queue cmdQueue;
    cl_program program;

    //std::vector<cl::Platform> platforms;
    //std::vector<std::vector<cl::Device>> devices;

    //cl::Context context;
    //cl::CommandQueue cmdQueue;
    //cl::Program program;

    ///Evento para controlar tempo gasto
    cl_ulong inicio, fim;
    cl_event e_tempo;

    cl_kernel testKernel;


    cl_kernel kernelCGP;
    //cl::Kernel kernelCGPDE;
    //cl::Kernel kernelDE;

    cl_kernel kernelTrain;
    cl_kernel kernelValid;
    cl_kernel kernelTest;

    cl_kernel kernelEvolve;

    cl_kernel kernelEvaluate;
    cl_kernel kernelEvaluate2;

    ///Buffers
    cl_mem bufferSeeds;

    cl_mem bufferDataOut;

    cl_mem bufferOutputsTrain;
    cl_mem bufferDatasetTrain;

    cl_mem bufferDatasetValid;
    cl_mem bufferOutputsValid;

    cl_mem bufferDatasetTest;
    cl_mem bufferOutputsTest;

    cl_mem bufferFunctions;

    cl_mem bufferBest;
    cl_mem bufferPopulation;

    size_t numPoints;
    size_t numPointsTrain;
    size_t numPointsValid;
    size_t numPointsTest;

    size_t maxLocalSize;

    size_t localSizeTrain;
    size_t localSizeValid;
    size_t localSizeTest;

    size_t globalSizeTrain;
    size_t globalSizeValid;
    size_t globalSizeTest;

    size_t localSizeAval;
    size_t globalSizeAval;

    size_t localSizeEvol;
    size_t globalSizeEvol;

    std::string compileFlags;

    cl_command_queue_properties commandQueueProperties;

    float* transposeDatasetOutput;

    float* transposeDatasetTrain;
    float* transposeOutputsTrain;

    float* transposeDatasetValid;
    float* transposeOutputsValid;

    float* transposeDatasetTest;
    float* transposeOutputsTest;


    void allocateBuffers(Parameters* p, int sizeTrain, int sizeValid, int sizeTest);
    void setNDRages();
    void setCompileFlags();
    std::string setProgramSource(Parameters* p, Dataset* fullData);
    void buildProgram(Parameters* p, Dataset* fullData, std::string sourceFileStr);
    void buildKernels();
    void writeReadOnlyBufers(Parameters* p, int* seeds);
    void transposeDatasets(Dataset* train, Dataset* valid, Dataset* test);

    void writeBestBuffer(Chromosome* best);
    void writePopulationBuffer(Chromosome* population);

    void readBestBuffer(Chromosome* best);
    void readPopulationBuffer(Chromosome* population);
    void readSeedsBuffer(int* seeds);

    void finishCommandQueue();

    void enqueueCGPKernel();

    void enqueueTestKernel();

    void enqueueEvolveKernel();

    void enqueueEvaluationKernel();
    void enqueueEvaluationKernel2();

    double getKernelElapsedTime();
    void printOpenclDeviceInfo();
private:

    void checkError(cl_int result);
    void transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs);
    void transposeDataOut(Dataset* data, float** transposeDatasetOutput);
    const char *getErrorString(cl_int error);

};


#endif //P_CGPDE_OCLCONFIG_H
