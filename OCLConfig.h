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

    std::vector<cl::Platform> platforms;
    std::vector<std::vector<cl::Device>> devices;

    std::vector<cl::ImageFormat> imageFormats;
    cl::size_t<3> origin;
    cl::size_t<3> imgSize;

    cl::Context context;
    cl::CommandQueue cmdQueue;
    cl::Program program;

    ///Evento para controlar tempo gasto
    cl_ulong inicio, fim;
    cl::Event e_tempo;

    cl::Kernel testKernel;


    cl::Kernel kernelCGP;
    //cl::Kernel kernelCGPDE;
    //cl::Kernel kernelDE;

    cl::Kernel kernelTrain;
    cl::Kernel kernelValid;
    cl::Kernel kernelTest;

    cl::Kernel kernelEvolve;

    cl::Kernel kernelEvaluate;
    cl::Kernel kernelEvaluateActive;
    cl::Kernel kernelEvaluateImage;
    cl::Kernel kernelEvaluateImageValidation;

    ///Buffers
    cl::Buffer bufferSeeds;

    cl::Buffer bufferDataOut;

    cl::Buffer bufferDatasetTrain;
    cl::Buffer bufferOutputsTrain;

    cl::Buffer bufferDatasetValid;
    cl::Buffer bufferOutputsValid;

    cl::Buffer bufferDatasetTest;
    cl::Buffer bufferOutputsTest;

    cl::Buffer bufferFunctions;

    cl::Buffer bufferBest;
    cl::Buffer bufferPopulation;
    cl::Buffer bufferPopulationActive;

    cl::Buffer bufferFitness;
    cl::Buffer bufferFitnessValidation;

    cl::Image2DArray populationImage;
    unsigned int* populationImageObject;

    cl::ImageFormat image_format;
    cl_image_desc image_desc;

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
    void writePopulationActiveBuffer(ActiveChromosome* population);

    void readBestBuffer(Chromosome* best);
    void readPopulationBuffer(Chromosome* population);
    void readPopulationActiveBuffer(ActiveChromosome* population);
    void readSeedsBuffer(int* seeds);
    void readFitnessBuffer(float* fitness);
    void readFitnessValidationBuffer(float* fitnessValidation);

    void setupImageBuffers();

    void finishCommandQueue();

    void enqueueCGPKernel();

    void enqueueTrainKernel();
    void enqueueValidationKernel();
    void enqueueTestKernel();


    void enqueueEvolveKernel();

    void enqueueEvaluationKernel();
    void enqueueEvaluationActiveKernel();
    void enqueueEvaluationImageKernel();
    void enqueueEvaluationImageValidationKernel();


    double getKernelElapsedTime();

    void writeImageBuffer(ActiveChromosome* population);

private:
    void printOpenclDeviceInfo();
    void checkError(cl_int result);
    void transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs);
    void transposeDataOut(Dataset* data, float** transposeDatasetOutput);
    const char *getErrorString(cl_int error);

};


#endif //P_CGPDE_OCLCONFIG_H
