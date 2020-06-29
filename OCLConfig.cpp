//
// Created by bruno on 25/06/2020.
//

#include "OCLConfig.h"


OCLConfig::OCLConfig() {

    int result = cl::Platform::get(&platforms);
    checkError(result);

    for(int i = 0; i < platforms.size(); i++){
        devices.emplace_back(std::vector<cl::Device>());
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices[i]);
    }

    context = cl::Context(devices[GPU_PLATFORM], nullptr, nullptr, nullptr, &result);
    checkError(result);

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

        }
    }
}

void OCLConfig::allocateBuffers(Parameters* p, int sizeTrain, int sizeValid, int sizeTest){
    int result;
    ///Buffers
    bufferSeeds = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_INDIV * sizeof(int), nullptr,  &result);
    checkError(result);

    bufferDatasetTrain = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTrain * p->N * sizeof(double), nullptr,  &result);
    checkError(result);

    bufferOutputsTrain = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTrain * p->O * sizeof(double), nullptr,  &result);
    checkError(result);

    bufferDatasetValid = cl::Buffer(context, CL_MEM_READ_ONLY, sizeValid * p->N * sizeof(double), nullptr,  &result);
    checkError(result);

    bufferOutputsValid = cl::Buffer(context, CL_MEM_READ_ONLY, sizeValid * p->O * sizeof(double), nullptr,  &result);
    checkError(result);

    bufferDatasetTest = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTest * p->N * sizeof(double), nullptr,  &result);
    checkError(result);

    bufferOutputsTest = cl::Buffer(context, CL_MEM_READ_ONLY, sizeTest * p->O * sizeof(double), nullptr,  &result);
    checkError(result);

    bufferFunctions = cl::Buffer(context, CL_MEM_READ_ONLY, ((p->NUM_FUNCTIONS)) * sizeof(unsigned int), nullptr,  &result);
    checkError(result);

    bufferBest = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Chromosome), nullptr,  &result);
    checkError(result);

    bufferPopulation = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_INDIV * sizeof(Chromosome), nullptr, &result);
    checkError(result);

    numPointsTrain = sizeTrain;
    numPointsValid = sizeValid;
    numPointsTest = sizeTest;
    numPoints = numPointsTrain > numPointsValid ? numPointsTrain : numPointsValid;
};

void OCLConfig::setNDRages() {
    //FOR GPU

    std::cout << "Definindo NDRanges ...";
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
    globalSizeTrain = localSizeTrain * 1;//* NUM_INDIV;
    globalSizeValid = localSizeValid * 1;//* NUM_INDIV;
    globalSizeTest = localSizeTest * 1;//* NUM_INDIV;
    globalSizeAval = localSizeAval * NUM_INDIV;


    ///estes tamanhos servem para a evolução treinamento+validação
    ///a ideia é que o kernel seja criado com esse tamanho padrão que englobe tanto os dados de
    ///treino quanto de validação
    //localSizeAval = localSizeTrain > localSizeValid ? localSizeTrain : localSizeValid;

    localSizeEvol = 1;
    globalSizeEvol = NUM_INDIV;

    std::cout << "...fim." << std::endl;

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
    double* transposeDataset;
    double* transposeOutput;
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
            "#define LOCAL_SIZE_TEST " + ToString( localSizeTest ) + "\n";// +
            //"__constant double constDataset[" + ToString(fullData->M * fullData->N) + "] = " + datasetString + "\n" +
            //"__constant double constOutputs[" + ToString(fullData->M * fullData->O) + "] = " + outputString + "\n";


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
    //kernelEvolve = cl::Kernel(program, "evolve", &result);
    //checkError(result);


    kernelTest = cl::Kernel(program, "evaluateTest", &result);
    checkError(result);
    kernelTest.setArg(0, bufferDatasetTest);
    kernelTest.setArg(1, bufferOutputsTest);
    kernelTest.setArg(2, bufferFunctions);
    kernelTest.setArg(3, bufferBest);
    kernelTest.setArg(4, (int)localSizeTest * sizeof(double), nullptr);

    kernelCGP = cl::Kernel(program, "CGP", &result);
    checkError(result);
    ///set fixed params
    kernelCGP.setArg(0, bufferDatasetTrain);
    kernelCGP.setArg(1, bufferOutputsTrain);
    kernelCGP.setArg(2, bufferDatasetValid);
    kernelCGP.setArg(3, bufferOutputsValid);
    kernelCGP.setArg(4, bufferFunctions);
    kernelCGP.setArg(5, bufferSeeds);
    kernelCGP.setArg(6, bufferPopulation);
    kernelCGP.setArg(7, bufferBest);
    kernelCGP.setArg(8, (int)localSizeAval * sizeof(double), nullptr);
}

void OCLConfig::transposeDatasets(Dataset* train, Dataset* valid, Dataset* test){
    std::cout << "Transpondo dados..." << std::endl;
    transposeData(train, &transposeDatasetTrain, &transposeOutputsTrain);
    transposeData(valid, &transposeDatasetValid, &transposeOutputsValid);
    transposeData(test, &transposeDatasetTest, &transposeOutputsTest);
}


void OCLConfig::writeReadOnlyBufers(Parameters* p, int* seeds){

    cmdQueue.enqueueWriteBuffer(bufferSeeds, CL_FALSE, 0, NUM_INDIV * sizeof(int), seeds);

    cmdQueue.enqueueWriteBuffer(bufferDatasetTrain, CL_FALSE, 0, numPointsTrain * p->N * sizeof(double), transposeDatasetTrain);
    cmdQueue.enqueueWriteBuffer(bufferOutputsTrain, CL_FALSE, 0, numPointsTrain * p->O * sizeof(double), transposeOutputsTrain);

    cmdQueue.enqueueWriteBuffer(bufferDatasetValid, CL_FALSE, 0, numPointsValid * p->N * sizeof(double), transposeDatasetValid);
    cmdQueue.enqueueWriteBuffer(bufferOutputsValid, CL_FALSE, 0, numPointsValid * p->O * sizeof(double), transposeOutputsValid);

    cmdQueue.enqueueWriteBuffer(bufferDatasetTest, CL_FALSE, 0, numPointsTest * p->N * sizeof(double), transposeDatasetTest);
    cmdQueue.enqueueWriteBuffer(bufferOutputsTest, CL_FALSE, 0, numPointsTest * p->O * sizeof(double), transposeOutputsTest);

    cmdQueue.enqueueWriteBuffer(bufferFunctions, CL_FALSE, 0, (p->NUM_FUNCTIONS) * sizeof(unsigned int), p->functionSet);

    cmdQueue.finish();
}



void OCLConfig::transposeData(Dataset* data, double** transposeDataset, double** transposeOutputs){

    (*transposeDataset) = new double [data->M * data->N];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    unsigned int pos = 0;
    //std::cout << "Transpondo dados..." << std::endl;
    for(int j = 0; j < data->N; ++j ){
        for(int i = 0; i < data->M; ++i ){
            (*transposeDataset)[pos++] = data->data[i][j];
        }
    }

    (*transposeOutputs) = new double [data->M * data->O];
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
    cmdQueue.enqueueWriteBuffer(bufferBest, CL_FALSE, 0, sizeof(Chromosome), best);
}

void OCLConfig::writePopulationBuffer(Chromosome* population){
    cmdQueue.enqueueWriteBuffer(bufferPopulation, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), population);
}

void OCLConfig::readBestBuffer(Chromosome* best){
    cmdQueue.enqueueReadBuffer(bufferBest, CL_FALSE, 0,  sizeof(Chromosome), best);
}

void OCLConfig::readPopulationBuffer(Chromosome* population){
    cmdQueue.enqueueReadBuffer(bufferPopulation, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), population);
}

void OCLConfig::readSeedsBuffer(int* seeds){
    cmdQueue.enqueueReadBuffer(bufferSeeds, CL_FALSE, 0, NUM_INDIV * sizeof(int), seeds);
}

void OCLConfig::finishCommandQueue(){
    cmdQueue.finish();
}

void OCLConfig::enqueueCGPKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelCGP, cl::NullRange, cl::NDRange(globalSizeAval), cl::NDRange(localSizeAval));
    checkError(result);
}

void OCLConfig::enqueueTestKernel(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelTest, cl::NullRange, cl::NDRange(globalSizeTest), cl::NDRange(localSizeTest));
    checkError(result);
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