//
// Created by bruno on 25/06/2020.
//

#include "OCLConfig.h"

void printImageFormat(cl_image_format format)
{
    #define CASE(order) case order: std::cout << #order; break;
    switch (format.image_channel_order)
    {
        CASE(CL_R);
        CASE(CL_A);
        CASE(CL_RG);
        CASE(CL_RA);
        CASE(CL_RGB);
        CASE(CL_RGBA);
        CASE(CL_BGRA);
        CASE(CL_ARGB);
        CASE(CL_INTENSITY);
        CASE(CL_LUMINANCE);
        CASE(CL_Rx);
        CASE(CL_RGx);
        CASE(CL_RGBx);
        CASE(CL_DEPTH);
        CASE(CL_DEPTH_STENCIL);
    }
    #undef CASE

    std::cout << " - ";

    #define CASE(type) case type: std::cout << #type; break;
    switch (format.image_channel_data_type)
    {
        CASE(CL_SNORM_INT8);
        CASE(CL_SNORM_INT16);
        CASE(CL_UNORM_INT8);
        CASE(CL_UNORM_INT16);
        CASE(CL_UNORM_SHORT_565);
        CASE(CL_UNORM_SHORT_555);
        CASE(CL_UNORM_INT_101010);
        CASE(CL_SIGNED_INT8);
        CASE(CL_SIGNED_INT16);
        CASE(CL_SIGNED_INT32);
        CASE(CL_UNSIGNED_INT8);
        CASE(CL_UNSIGNED_INT16);
        CASE(CL_UNSIGNED_INT32);
        CASE(CL_HALF_FLOAT);
        CASE(CL_FLOAT);
        CASE(CL_UNORM_INT24);
    }
    #undef CASE

    std::cout << std::endl;
}

OCLConfig::OCLConfig() {

    int result = cl::Platform::get(&platforms);
    checkError(result);

    for(int i = 0; i < platforms.size(); i++){
        devices.emplace_back(std::vector<cl::Device>());
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices[i]);
    }

    context = cl::Context(devices[GPU_PLATFORM], nullptr, nullptr, nullptr, &result);
    checkError(result);

    context.getSupportedImageFormats(CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, &imageFormats);
    for (int i = 0; i < imageFormats.size(); ++i) {
        printImageFormat(imageFormats[i]);
    }
    commandQueueProperties = CL_QUEUE_PROFILING_ENABLE;
    cmdQueue = cl::CommandQueue(context, devices[GPU_PLATFORM][GPU_DEVICE], commandQueueProperties, &result);
    checkError(result);

    maxLocalSize = cmdQueue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    printOpenclDeviceInfo();
}

void OCLConfig::checkError(cl_int result){
    if(result != CL_SUCCESS)
        std::cerr << getErrorString(result) << std::endl;
}
void OCLConfig::printOpenclDeviceInfo(){
    for(int j = 0; j < platforms.size(); ++j) {
        std::cout << "Available Devices for Platform " << platforms[j].getInfo<CL_PLATFORM_NAME>() << ":\n";

        for (int i = 0; i < devices[j].size(); ++i) {
            std::cout << "[" << i << "]" << devices[j][i].getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "\tType:   " << devices[j][i].getInfo<CL_DEVICE_TYPE>() << std::endl;
            std::cout << "\tOpenCL: " << devices[j][i].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
            std::cout << "\tMax Comp Un: " << devices[j][i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\tMax WrkGr Sz: " << devices[j][i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "\tFp config: " << devices[j][i].getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() << std::endl;
            std::cout << "\tMax Mem Alloc: " << devices[j][i].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
            std::cout << "\tLocal Mem Size: " << devices[j][i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
            std::cout << "\tMax Const Size: " << devices[j][i].getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << std::endl;
            std::cout << "\tDevice Extensions: " << devices[j][i].getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
            std::cout << "\tDevice Img Support: " << devices[j][i].getInfo<CL_DEVICE_IMAGE_SUPPORT>() << std::endl;


        }
    }
    std::cout << "\tNode size: " << sizeof(Node) << std::endl;
    std::cout << "\tIndividuals size: " << sizeof(Chromosome) << std::endl;
    std::cout <<  sizeof(int) <<sizeof(unsigned int) << sizeof(float) <<sizeof(double) <<std::endl;

}

void OCLConfig::allocateBuffers(Parameters* p, int sizeTrain, int sizeValid, int sizeTest){
    int result;
    std::cout << "Allocating buffers... " << std::endl;

    ///Buffers
    bufferSeeds = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_INDIV * maxLocalSize  * sizeof(int), nullptr,  &result);
    checkError(result);

    bufferDataOut      = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTrain * (p->N + p->O) * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferDatasetTrain = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTrain * p->N * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferOutputsTrain = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTrain * p->O * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferDatasetValid = cl::Buffer(context, CL_MEM_READ_ONLY, sizeValid * p->N * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferOutputsValid = cl::Buffer(context, CL_MEM_READ_ONLY, sizeValid * p->O * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferDatasetTest = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTest * p->N * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferOutputsTest = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTest * p->O * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferFunctions = cl::Buffer(context, CL_MEM_READ_ONLY, ((p->NUM_FUNCTIONS)) * sizeof(unsigned int), nullptr,  &result);
    checkError(result);

    bufferBest = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Chromosome), nullptr,  &result);
    checkError(result);

    bufferPopulation = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_INDIV * sizeof(Chromosome), nullptr, &result);
    checkError(result);

    bufferPopulationActive = cl::Buffer(context, CL_MEM_READ_ONLY, NUM_INDIV * sizeof(ActiveChromosome), nullptr, &result);
    checkError(result);

    bufferFitness =  cl::Buffer(context, CL_MEM_WRITE_ONLY, NUM_INDIV * sizeof(float), nullptr, &result);
    checkError(result);

    bufferFitnessValidation =  cl::Buffer(context, CL_MEM_WRITE_ONLY, NUM_INDIV * sizeof(float), nullptr, &result);
    checkError(result);

    numPointsTrain = sizeTrain;
    numPointsValid = sizeValid;
    numPointsTest = sizeTest;
    numPoints = numPointsTrain > numPointsValid ? numPointsTrain : numPointsValid;

    (transposeDatasetTrain) = new float [numPointsTrain * p->N];
    (transposeOutputsTrain) = new float [numPointsTrain * p->O];

    (transposeDatasetValid) = new float [numPointsValid * p->N];
    (transposeOutputsValid) = new float [numPointsValid * p->O];

    (transposeDatasetTest) = new float [numPointsTest * p->N];
    (transposeOutputsTest) = new float [numPointsTest * p->O];

    (transposeDatasetOutput) = new float [numPoints * (p->N + p->O)];
}

void OCLConfig::setNDRages() {
    //FOR GPU

    std::cout << "Setting NDRanges ..." << std::endl;
    if(numPointsTrain < maxLocalSize){
        localSizeTrain = numPointsTrain;
    }else{
        localSizeTrain = maxLocalSize;
    }

    if(numPointsValid < maxLocalSize){
        localSizeValid = numPointsValid;
    }else{
        localSizeValid = maxLocalSize;
    }

    if(numPointsTest < maxLocalSize){
        localSizeTest = numPointsTest;
    }else{
        localSizeTest = maxLocalSize;
    }

    if(numPoints < maxLocalSize){
        localSizeAval = numPoints;
    }else{
        localSizeAval = maxLocalSize;
    }

    // One individual per work-group
    globalSizeTrain = localSizeTrain * NUM_INDIV;
    globalSizeValid = localSizeValid * NUM_INDIV;
    globalSizeTest = localSizeTest * 1;//* NUM_INDIV;
    globalSizeAval = localSizeAval * NUM_INDIV;


    ///estes tamanhos servem para a evolução treinamento+validação
    ///a ideia é que o kernel seja criado com esse tamanho padrão que englobe tanto os dados de
    ///treino quanto de validação
    //localSizeAval = localSizeTrain > localSizeValid ? localSizeTrain : localSizeValid;

    if(MAX_NODES * MAX_ARITY < maxLocalSize){
        localSizeEvol = MAX_NODES * MAX_ARITY;
    }else{
        localSizeEvol = maxLocalSize;
    }

    //localSizeEvol = MAX_NODES * MAX_ARITY;
    globalSizeEvol = localSizeEvol * NUM_INDIV;


}

void OCLConfig::setCompileFlags(){

    std::cout << "Setting compile flags..." << std::endl;

    if( !IsPowerOf2( localSizeAval ) )
        compileFlags += " -D LOCAL_SIZE_IS_NOT_POWER_OF_2";
    if( !IsPowerOf2( localSizeTrain ) )
        compileFlags += " -D LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2";
    if( !IsPowerOf2( localSizeValid ) )
        compileFlags += " -D LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2";
    if( !IsPowerOf2( localSizeTest ) )
        compileFlags += " -D LOCAL_SIZE_TEST_IS_NOT_POWER_OF_2";

    if( numPoints % (localSizeAval) != 0 )
        compileFlags += " -D NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE";
    if( numPointsValid % (localSizeAval) != 0 )
        compileFlags += " -D NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL";

    if( numPointsTrain % (localSizeTrain) != 0 )
        compileFlags += " -D NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE";
    if( numPointsValid % (localSizeValid) != 0 )
        compileFlags += " -D NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE";
    if( numPointsTest % (localSizeTest) != 0 )
        compileFlags += " -D NUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE";

    compileFlags += " -D LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeAval) );
    compileFlags += " -D LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeTrain) );
    compileFlags += " -D LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeValid) );
    compileFlags += " -D LOCAL_SIZE_TEST_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeTest) );

}

std::string OCLConfig::setProgramSource(Parameters* p, Dataset* fullData){
    float* transposeDataset = new float [fullData->M * fullData->N];
    float* transposeOutput = new float [fullData->M * fullData->O];
    transposeData(fullData, &transposeDataset, &transposeOutput);

    std::string datasetString = "{";
    int i = 0;
    for(i = 0; i < fullData->M-1; i++){
        datasetString+= ToString(transposeDataset[i]) + ", ";
    }
    datasetString+= ToString(transposeDataset[i]);
    datasetString += "};";

    std::string outputString = "{";
    i = 0;
    for(i = 0; i < fullData->M-1; i++){
        outputString+= ToString(transposeOutput[i]) + ", ";
    }
    outputString+= ToString(transposeOutput[i]);
    outputString += "};";

    std::string program_src =
            "#define SEED "  + ToString( SEED ) + "\n" +
            "#define N   " + ToString( p->N ) + "\n" +
            "#define O   " + ToString( p->O ) + "\n" +
            "#define M   " + ToString( numPoints ) + "\n" +
            "#define M_TRAIN   " + ToString( numPointsTrain ) + "\n" +
            "#define M_VALIDATION   " + ToString( numPointsValid ) + "\n" +
            "#define M_TEST   " + ToString( numPointsTest ) + "\n" +
            "#define WEIGTH_RANGE    " + ToString(p->weightRange) + "\n" +
            "#define NUM_FUNCTIONS    " + ToString(p->NUM_FUNCTIONS) + "\n" +
            "#define SIG    " + ToString(SIG) + "\n" +
            "#define MAX_NODES     " + ToString( MAX_NODES ) + "\n" +
            "#define MAX_OUTPUTS  " + ToString( MAX_OUTPUTS ) + "\n" +
            "#define NUM_INDIV   " + ToString( NUM_INDIV ) + "\n" +
            "#define MAX_ARITY "    + ToString( MAX_ARITY ) + "\n" +
            "#define PROB_CROSS  "+ ToString( PROB_CROSS ) + "\n" +
            "#define PROB_MUT    "+ ToString( PROB_MUT ) + "\n" +
            "#define NUM_GENERATIONS    "+ ToString( NUM_GENERATIONS ) + "\n" +
            "#define CONST_PI    "+ ToString( CONST_PI ) + "\n" +
            "#define LOCAL_SIZE " + ToString( localSizeAval ) + "\n"+
            "#define LOCAL_SIZE_TRAIN " + ToString( localSizeTrain ) + "\n"+
            "#define LOCAL_SIZE_VALIDATION " + ToString( localSizeValid ) + "\n"+
            "#define LOCAL_SIZE_TEST " + ToString( localSizeTest ) + "\n" +
            "#define LOCAL_SIZE_EVOL " + ToString( localSizeEvol ) + "\n";// +
            //"__constant float constDataset[" + ToString(fullData->M * fullData->N) + "] = " + datasetString + "\n" +
            //"__constant float constOutputs[" + ToString(fullData->M * fullData->O) + "] = " + outputString + "\n";

    delete [] transposeDataset;
    delete [] transposeOutput;

    return program_src;
}

void OCLConfig::buildProgram(Parameters* p, Dataset* fullData, std::string sourceFileStr){
    ///Program build
    std::ifstream sourceFileName(sourceFileStr);//"kernels\\kernel_split_data.cl");
    std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName),(std::istreambuf_iterator<char>()));

    std::string program_src = setProgramSource(p,fullData) + sourceFile;
    //std::cout << program_src << std::endl;
    program = cl::Program(context, program_src);

    int result = program.build(devices[GPU_PLATFORM], compileFlags.c_str());

    if(result != CL_SUCCESS){
        std::cerr << getErrorString(result) << std::endl;
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[GPU_PLATFORM][GPU_DEVICE]);
        std::cerr << "Build log:" << std::endl
                  << buildLog << std::endl;
    }
}

void OCLConfig::buildKernels(){
    int result;
    ///Kernels


    kernelEvaluate = cl::Kernel(program, "evaluateTrainValidation", &result);
    checkError(result);
    kernelEvaluate.setArg(0, bufferDatasetTrain);
    kernelEvaluate.setArg(1, bufferOutputsTrain);
    kernelEvaluate.setArg(2, bufferDatasetValid);
    kernelEvaluate.setArg(3, bufferOutputsValid);
    kernelEvaluate.setArg(4, bufferFunctions);
    kernelEvaluate.setArg(5, bufferPopulation);
    kernelEvaluate.setArg(6, bufferBest);
    kernelEvaluate.setArg(7, (int)localSizeAval * sizeof(float), nullptr);


    kernelEvaluateActive = cl::Kernel(program, "evaluateTrainValidationActive", &result);
    checkError(result);
    kernelEvaluateActive.setArg(0, bufferDatasetTrain);
    kernelEvaluateActive.setArg(1, bufferOutputsTrain);
    kernelEvaluateActive.setArg(2, bufferDatasetValid);
    kernelEvaluateActive.setArg(3, bufferOutputsValid);
    kernelEvaluateActive.setArg(4, bufferFunctions);
    kernelEvaluateActive.setArg(5, bufferPopulationActive);
    kernelEvaluateActive.setArg(6, (int)localSizeAval * sizeof(float), nullptr);
    kernelEvaluateActive.setArg(7, bufferFitness);
    kernelEvaluateActive.setArg(8, bufferFitnessValidation);

    kernelEvaluateImage = cl::Kernel(program, "evaluateTrainImage", &result);
    checkError(result);
    kernelEvaluateImage.setArg(0, bufferDatasetTrain);
    kernelEvaluateImage.setArg(1, bufferOutputsTrain);
    kernelEvaluateImage.setArg(2, bufferFunctions);
    kernelEvaluateImage.setArg(3, populationImage);
    kernelEvaluateImage.setArg(4, (int)localSizeTrain * sizeof(float), nullptr);
    kernelEvaluateImage.setArg(5, bufferFitness);
    kernelEvaluateImage.setArg(6, bufferFitnessValidation);

    kernelEvaluateImageValidation = cl::Kernel(program, "evaluateValidationImage", &result);
    checkError(result);
    kernelEvaluateImageValidation.setArg(0, bufferDatasetValid);
    kernelEvaluateImageValidation.setArg(1, bufferOutputsValid);
    kernelEvaluateImageValidation.setArg(2, bufferFunctions);
    kernelEvaluateImageValidation.setArg(3, populationImage);
    kernelEvaluateImageValidation.setArg(4, (int)localSizeValid * sizeof(float), nullptr);
    kernelEvaluateImageValidation.setArg(5, bufferFitness);
    kernelEvaluateImageValidation.setArg(6, bufferFitnessValidation);


    kernelTrain = cl::Kernel(program, "evaluateTrain", &result);
    checkError(result);
    kernelTrain.setArg(0, bufferDatasetTrain);
    kernelTrain.setArg(1, bufferOutputsTrain);
    kernelTrain.setArg(2, bufferFunctions);
    kernelTrain.setArg(3, bufferPopulation);
    kernelTrain.setArg(4, (int)localSizeTrain * sizeof(float), nullptr);

    kernelValid = cl::Kernel(program, "evaluateValidation", &result);
    checkError(result);
    kernelValid.setArg(0, bufferDatasetValid);
    kernelValid.setArg(1, bufferOutputsValid);
    kernelValid.setArg(2, bufferFunctions);
    kernelValid.setArg(3, bufferPopulation);
    kernelValid.setArg(4, (int)localSizeValid * sizeof(float), nullptr);

    kernelTest = cl::Kernel(program, "evaluateTest", &result);
    checkError(result);
    kernelTest.setArg(0, bufferDatasetTest);
    kernelTest.setArg(1, bufferOutputsTest);
    kernelTest.setArg(2, bufferFunctions);
    kernelTest.setArg(3, bufferBest);
    kernelTest.setArg(4, (int)localSizeTest * sizeof(float), nullptr);


    kernelEvolve = cl::Kernel(program, "evolve", &result);
    checkError(result);
    kernelEvolve.setArg(0, bufferFunctions);
    kernelEvolve.setArg(1, bufferSeeds);
    kernelEvolve.setArg(2, bufferPopulation);
    kernelEvolve.setArg(3, bufferBest);

}

void OCLConfig::transposeDatasets(Dataset* train, Dataset* valid, Dataset* test){
    std::cout << "Transpondo dados..." << std::endl;
    transposeData(train, &transposeDatasetTrain, &transposeOutputsTrain);
    transposeData(valid, &transposeDatasetValid, &transposeOutputsValid);
    transposeData(test, &transposeDatasetTest, &transposeOutputsTest);
    transposeDataOut(train, &transposeDatasetOutput);
}

void OCLConfig::writeReadOnlyBufers(Parameters* p, int* seeds){
    int result;
    result = cmdQueue.enqueueWriteBuffer(bufferSeeds, CL_FALSE, 0, NUM_INDIV * maxLocalSize  * sizeof(int), seeds);
    checkError(result);

    cmdQueue.enqueueWriteBuffer(bufferDataOut, CL_FALSE, 0, numPoints * (p->N + p->O) * sizeof(float), transposeDatasetOutput);

    cmdQueue.enqueueWriteBuffer(bufferDatasetTrain, CL_FALSE, 0, numPointsTrain * p->N * sizeof(float), transposeDatasetTrain);
    cmdQueue.enqueueWriteBuffer(bufferOutputsTrain, CL_FALSE, 0, numPointsTrain * p->O * sizeof(float), transposeOutputsTrain);

    cmdQueue.enqueueWriteBuffer(bufferDatasetValid, CL_FALSE, 0, numPointsValid * p->N * sizeof(float), transposeDatasetValid);
    cmdQueue.enqueueWriteBuffer(bufferOutputsValid, CL_FALSE, 0, numPointsValid * p->O * sizeof(float), transposeOutputsValid);

    cmdQueue.enqueueWriteBuffer(bufferDatasetTest, CL_FALSE, 0, numPointsTest * p->N * sizeof(float), transposeDatasetTest);
    cmdQueue.enqueueWriteBuffer(bufferOutputsTest, CL_FALSE, 0, numPointsTest * p->O * sizeof(float), transposeOutputsTest);

    cmdQueue.enqueueWriteBuffer(bufferFunctions, CL_FALSE, 0, (p->NUM_FUNCTIONS) * sizeof(unsigned int), p->functionSet);

    cmdQueue.finish();
}

void OCLConfig::transposeDataOut(Dataset* data, float** transposeDatasetOut){

    //(*transposeDatasetOut) = new float [data->M * (data->N + data->O)];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    unsigned int pos = 0;
    //std::cout << "Transpondo dados..." << std::endl;
    for(int j = 0; j < data->N + data->O; ++j ){
        for(int i = 0; i < data->M; ++i ){
            if(j < data->N){
                (*transposeDatasetOut)[pos++] = data->data[i][j];
                //std::cout << (*transposeDatasetOut)[pos-1] << " ";
            }else{
                (*transposeDatasetOut)[pos++] = data->output[i][j - data->N];
                //std::cout << (*transposeDatasetOut)[pos-1] << " ";
            }

        }
        //std::cout << std::endl;
    }

}

void OCLConfig::transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs){

    //(*transposeDataset) = new float [data->M * data->N];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    unsigned int pos = 0;
    //std::cout << "Transpondo dados..." << std::endl;
    for(int j = 0; j < data->N; ++j ){
        for(int i = 0; i < data->M; ++i ){
            (*transposeDataset)[pos++] = data->data[i][j];
        }
    }

    //(*transposeOutputs) = new float [data->M * data->O];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    pos = 0;
    //std::cout << "Transpondo outputs..." << std::endl;
    for(int j = 0; j < data->O; ++j ){
        for(int i = 0; i < data->M; ++i ){
            (*transposeOutputs)[pos++] = data->output[i][j];
        }
    }

}

void OCLConfig::writeBestBuffer(Chromosome* best){
    int result = cmdQueue.enqueueWriteBuffer(bufferBest, CL_FALSE, 0, sizeof(Chromosome), best);
    checkError(result);
}

void OCLConfig::writePopulationBuffer(Chromosome* population){
    int result = cmdQueue.enqueueWriteBuffer(bufferPopulation, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), population);
    checkError(result);
}

void OCLConfig::writePopulationActiveBuffer(ActiveChromosome* population){
    int result = cmdQueue.enqueueWriteBuffer(bufferPopulationActive, CL_FALSE, 0, NUM_INDIV * sizeof(ActiveChromosome), population);
    checkError(result);
}

void OCLConfig::readBestBuffer(Chromosome* best){
    int result = cmdQueue.enqueueReadBuffer(bufferBest, CL_FALSE, 0,  sizeof(Chromosome), best);
    checkError(result);
}

void OCLConfig::readPopulationBuffer(Chromosome* population){
    int result = cmdQueue.enqueueReadBuffer(bufferPopulation, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), population);
    checkError(result);
}

void OCLConfig::readPopulationActiveBuffer(ActiveChromosome* population){
    int result = cmdQueue.enqueueReadBuffer(bufferPopulationActive, CL_FALSE, 0, NUM_INDIV * sizeof(ActiveChromosome), population);
    checkError(result);
}

void OCLConfig::readFitnessBuffer(float* fitness){
    int result = cmdQueue.enqueueReadBuffer(bufferFitness, CL_FALSE, 0, NUM_INDIV * sizeof(float), fitness);
    checkError(result);
}

void OCLConfig::readFitnessValidationBuffer(float* fitnessValidation){
    int result = cmdQueue.enqueueReadBuffer(bufferFitnessValidation, CL_FALSE, 0, NUM_INDIV * sizeof(float), fitnessValidation);
    checkError(result);
}

void OCLConfig::readSeedsBuffer(int* seeds){
    int result = cmdQueue.enqueueReadBuffer(bufferSeeds, CL_FALSE, 0, NUM_INDIV * maxLocalSize  * sizeof(int), seeds);
    checkError(result);
}

void OCLConfig::finishCommandQueue(){
    int result = cmdQueue.finish();
    checkError(result);
}

void OCLConfig::enqueueCGPKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelCGP, cl::NullRange, cl::NDRange(globalSizeAval), cl::NDRange(localSizeAval),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueTrainKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelTrain, cl::NullRange, cl::NDRange(globalSizeTrain), cl::NDRange(localSizeTrain),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueValidationKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelValid, cl::NullRange, cl::NDRange(globalSizeValid), cl::NDRange(localSizeValid),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueTestKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelTest, cl::NullRange, cl::NDRange(globalSizeTest), cl::NDRange(localSizeTest),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueEvolveKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelEvolve, cl::NullRange, cl::NDRange(globalSizeEvol), cl::NDRange(localSizeEvol),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueEvaluationKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelEvaluate, cl::NullRange, cl::NDRange(globalSizeAval), cl::NDRange(localSizeAval),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueEvaluationActiveKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelEvaluateActive, cl::NullRange, cl::NDRange(globalSizeAval), cl::NDRange(localSizeAval),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueEvaluationImageKernel(){
    kernelEvaluateImage.setArg(3, populationImage);
    int result = cmdQueue.enqueueNDRangeKernel(kernelEvaluateImage, cl::NullRange, cl::NDRange(globalSizeTrain), cl::NDRange(localSizeTrain),
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueEvaluationImageValidationKernel() {
    kernelEvaluateImageValidation.setArg(3, populationImage);
    int result = cmdQueue.enqueueNDRangeKernel(kernelEvaluateImageValidation, cl::NullRange, cl::NDRange(globalSizeValid), cl::NDRange(localSizeValid),
                                               nullptr, &e_tempo);
    checkError(result);
}


void OCLConfig::setupImageBuffers(){
    image_format.image_channel_data_type = CL_UNSIGNED_INT16;
    image_format.image_channel_order = CL_RG;

    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = (MAX_ARITY + 1);
    image_desc.image_height = (MAX_NODES * 2) + 2;
    image_desc.image_depth = 0;
    image_desc.image_array_size = NUM_INDIV;

    image_desc.image_row_pitch = 0;
    image_desc.image_slice_pitch = 0;
    image_desc.num_mip_levels = 0;
    image_desc.num_samples = 0;
    image_desc.buffer = nullptr;

    int result;
    populationImage = cl::Image2DArray(context, CL_MEM_READ_ONLY,
            image_format,
            image_desc.image_array_size,
            image_desc.image_width,
            image_desc.image_height,
            0,
            0,
            nullptr, &result);
    checkError(result);


    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    imgSize[0] =static_cast<size_t>(image_desc.image_width);
    imgSize[1] =static_cast<size_t>(image_desc.image_height);
    imgSize[2] =static_cast<size_t>(image_desc.image_array_size);

    //populationImageObject = new unsigned int[imgSize[0]*imgSize[1]*imgSize[2]];

    populationImageObject = new unsigned int[image_desc.image_width *
                                            image_desc.image_height *
                                            image_desc.image_array_size];

}

void OCLConfig::writeImageBuffer(ActiveChromosome* population){

    int index = 0;

    for(int i = 0; i < NUM_INDIV; i++ ){
        populationImageObject[index++] = population[i].numActiveNodes;

        while(index % image_desc.image_width != 0) index++;

        for(int j = 0; j < MAX_NODES; j++){
            populationImageObject[index++] = population[i].nodes[j].originalIndex;
            for(int k = 0; k < MAX_ARITY; k++){
                populationImageObject[index++] = population[i].nodes[j].inputs[k];
            }
            populationImageObject[index++] = population[i].nodes[j].function;
            for(int k = 0; k < MAX_ARITY; k++){
                populationImageObject[index++] = *(unsigned int*)&population[i].nodes[j].inputsWeight[k];
            }
        }
        for(int j = 0; j < MAX_OUTPUTS; j++){
            populationImageObject[index++] = population[i].output[j];
        }

        while(index % image_desc.image_width != 0) index++;
        //std::cout << index << std::endl;
    }

    int result = cmdQueue.enqueueWriteImage( populationImage, CL_TRUE, origin, imgSize,0, 0, populationImageObject);
    checkError(result);
}

double OCLConfig::getKernelElapsedTime(){
    e_tempo.getProfilingInfo(CL_PROFILING_COMMAND_START, &inicio);
    e_tempo.getProfilingInfo(CL_PROFILING_COMMAND_END, &fim);
    return ((fim-inicio)/1.0E9);
}

const char* OCLConfig::getErrorString(cl_int error) {
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}