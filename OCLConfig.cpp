//
// Created by bruno on 25/06/2020.
//

#include "OCLConfig.h"


OCLConfig::OCLConfig() {

    int result = clGetPlatformIDs(0, NULL, &platformCount);
    checkError(result);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    result = clGetPlatformIDs(platformCount, platforms, nullptr);

    //int result = cl::Platform::get(&platforms);
    checkError(result);

    devices = (cl_device_id**) malloc(sizeof(cl_device_id*) * platformCount);
    deviceCount = (cl_uint*) malloc(sizeof(cl_uint) * platformCount);
    for(int i = 0; i < platformCount; i++){
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount[i]);
        devices[i] = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount[i]);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount[i], devices[i], nullptr);

        //devices.emplace_back(std::vector<cl::Device>());
        //platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices[i]);
    }

    context = clCreateContext(nullptr, 1, devices[GPU_PLATFORM], nullptr, nullptr, &result);

    //context = cl::Context(devices[GPU_PLATFORM], nullptr, nullptr, nullptr, &result);
    checkError(result);

    commandQueueProperties = CL_QUEUE_PROFILING_ENABLE;
    cmdQueue = clCreateCommandQueue(context, devices[GPU_PLATFORM][GPU_DEVICE], commandQueueProperties, &result);

    //cmdQueue = cl::CommandQueue(context, devices[GPU_PLATFORM][GPU_DEVICE], commandQueueProperties, &result);
    checkError(result);

    size_t valueSize;
    clGetDeviceInfo(devices[GPU_PLATFORM][GPU_DEVICE], CL_DEVICE_VERSION, 0, nullptr, &valueSize);

    result = clGetDeviceInfo(devices[GPU_PLATFORM][GPU_DEVICE], CL_DEVICE_MAX_WORK_GROUP_SIZE, valueSize, &maxLocalSize,
                             nullptr);
            //cmdQueue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    checkError(result);

    printOpenclDeviceInfo();
}

void OCLConfig::checkError(cl_int result){
    if(result != CL_SUCCESS)
        std::cerr << getErrorString(result) << std::endl;
}
void OCLConfig::printOpenclDeviceInfo(){
    for(int j = 0; j < platformCount; ++j) {
        char* value;
        size_t valueSize;

        clGetPlatformInfo(platforms[j], CL_PLATFORM_NAME, 0, nullptr, &valueSize);
        value = (char*) malloc(valueSize);
        clGetPlatformInfo(platforms[j], CL_PLATFORM_NAME, valueSize, value, nullptr);
        std::cout << "Available Devices for Platform " <<value << ":\n";
        free(value);

        for (int i = 0; i < deviceCount[j]; ++i) {


            clGetDeviceInfo(devices[j][i], CL_DEVICE_NAME, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_NAME, valueSize, value, nullptr);
            std::cout << "[" << i << "]" << value << std::endl;
            free(value);


            clGetDeviceInfo(devices[j][i], CL_DEVICE_TYPE, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_TYPE, valueSize, value, nullptr);
            std::cout << "\tType:   " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, nullptr);
            std::cout << "\tOpenCL: " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_COMPUTE_UNITS, valueSize, value, nullptr);
            std::cout << "\tMax Comp Un: " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_WORK_GROUP_SIZE, valueSize, value, nullptr);
            std::cout << "\tMax WrkGr Sz: " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_SINGLE_FP_CONFIG, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_SINGLE_FP_CONFIG, valueSize, value, nullptr);
            std::cout << "\tFp config: " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, valueSize, value, nullptr);
            std::cout << "\tMax Mem Alloc: " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_LOCAL_MEM_SIZE, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_LOCAL_MEM_SIZE, valueSize, value, nullptr);
            std::cout << "\tLocal Mem Size: " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, valueSize, value, nullptr);
            std::cout << "\tMax Const Size: " << value << std::endl;
            free(value);

            clGetDeviceInfo(devices[j][i], CL_DEVICE_EXTENSIONS, 0, nullptr, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j][i], CL_DEVICE_EXTENSIONS, valueSize, value, nullptr);
            std::cout << "\tDevice Extensions: " << value << std::endl;
            free(value);

        }
    }
}

void OCLConfig::allocateBuffers(Parameters* p, int sizeTrain, int sizeValid, int sizeTest){
    int result;
    std::cout << "Allocating buffers... " << std::endl;

    ///Buffers
    bufferSeeds = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_INDIV * maxLocalSize  * sizeof(int), nullptr,  &result);
    checkError(result);

    bufferDataOut      = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeTrain * (p->N + p->O) * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferDatasetTrain = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeTrain * p->N * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferOutputsTrain = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeTrain * p->O * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferDatasetValid = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeValid * p->N * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferOutputsValid = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeValid * p->O * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferDatasetTest = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeTest * p->N * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferOutputsTest = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeTest * p->O * sizeof(float), nullptr,  &result);
    checkError(result);

    bufferFunctions = clCreateBuffer(context, CL_MEM_READ_ONLY, ((p->NUM_FUNCTIONS)) * sizeof(unsigned int), nullptr,  &result);
    checkError(result);

    bufferBest = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Chromosome), nullptr,  &result);
    checkError(result);

    bufferPopulation = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_INDIV * sizeof(Chromosome), nullptr, &result);
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
    globalSizeTrain = localSizeTrain * 1;//* NUM_INDIV;
    globalSizeValid = localSizeValid * 1;//* NUM_INDIV;
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

#define MAX_SOURCE_SIZE (0x100000)

void OCLConfig::buildProgram(Parameters* p, Dataset* fullData, std::string sourceFileStr){
    ///Program build
    char *source_str;
    size_t source_size;
    FILE *fp;
    fp = fopen(sourceFileStr.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    //std::ifstream sourceFileName(sourceFileStr);//"kernels\\kernel_split_data.cl");

    //std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName),(std::istreambuf_iterator<char>()));

    std::string program_src = setProgramSource(p,fullData) + source_str;
    //std::cout << program_src << std::endl;
    // declaring character array
    char char_array[program_src.size() + 1];
    source_size = program_src.size() + 1;
    strncpy(char_array, program_src.c_str(), source_size);

    int result;

    program = clCreateProgramWithSource(context, 1, (const char **)&char_array, (const size_t *)&source_size, &result);
    checkError(result);

    result = clBuildProgram(program, 1, devices[GPU_PLATFORM], compileFlags.c_str(), nullptr, nullptr);
    checkError(result);

    if(result != CL_SUCCESS){
        std::cerr << getErrorString(result) << std::endl;
        char* value;
        size_t valueSize;
        clGetProgramBuildInfo(program, devices[GPU_PLATFORM][GPU_DEVICE], CL_PROGRAM_BUILD_LOG, 0, nullptr, &valueSize);
        value = (char*) malloc(valueSize);
        clGetProgramBuildInfo(program, devices[GPU_PLATFORM][GPU_DEVICE], CL_PROGRAM_BUILD_LOG, valueSize, value,
                              nullptr);
        //std::string buildLog = clGetProgramBuildInfo(program, devices[GPU_PLATFORM])
        //program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[GPU_PLATFORM][GPU_DEVICE]);
        std::cerr << "Build log:" << std::endl
                  << value << std::endl;
        free(value);
    }
}

void OCLConfig::buildKernels(){
    int result;
    ///Kernels


    kernelEvaluate = clCreateKernel(program, "evaluateTrainValidation", &result);
    checkError(result);
    clSetKernelArg(kernelEvaluate, 0, sizeof(cl_mem), bufferDatasetTrain);
    clSetKernelArg(kernelEvaluate, 1, sizeof(cl_mem), bufferOutputsTrain);
    clSetKernelArg(kernelEvaluate, 2, sizeof(cl_mem), bufferDatasetValid);
    clSetKernelArg(kernelEvaluate, 3, sizeof(cl_mem), bufferOutputsValid);
    clSetKernelArg(kernelEvaluate, 4, sizeof(cl_mem), bufferFunctions);
    clSetKernelArg(kernelEvaluate, 5, sizeof(cl_mem), bufferPopulation);
    clSetKernelArg(kernelEvaluate, 6, sizeof(cl_mem), bufferBest);
    clSetKernelArg(kernelEvaluate, 7, (int)localSizeAval * sizeof(float), nullptr);

    /*
    kernelEvaluate2 = cl::Kernel(program, "evaluate2", &result);
    checkError(result);
    kernelEvaluate2.setArg(0, bufferDataOut);
    kernelEvaluate2.setArg(1, bufferFunctions);
    kernelEvaluate2.setArg(2, bufferPopulation);
    kernelEvaluate2.setArg(3, (int)localSizeAval * sizeof(float), nullptr);
*/

    kernelTest = clCreateKernel(program, "evaluateTest", &result);
    checkError(result);
    clSetKernelArg(kernelTest, 0, sizeof(cl_mem), bufferDatasetTest);
    clSetKernelArg(kernelTest, 1, sizeof(cl_mem), bufferOutputsTest);
    clSetKernelArg(kernelTest, 2, sizeof(cl_mem), bufferFunctions);
    clSetKernelArg(kernelTest, 3, sizeof(cl_mem), bufferBest);
    clSetKernelArg(kernelTest, 4, (int)localSizeTest * sizeof(float), nullptr);

    kernelCGP = clCreateKernel(program, "CGP", &result);
    checkError(result);
    ///set fixed params
    clSetKernelArg(kernelCGP, 0,sizeof(cl_mem), bufferDatasetTrain);
    clSetKernelArg(kernelCGP, 1,sizeof(cl_mem), bufferOutputsTrain);
    clSetKernelArg(kernelCGP, 2,sizeof(cl_mem), bufferDatasetValid);
    clSetKernelArg(kernelCGP, 3,sizeof(cl_mem), bufferOutputsValid);
    clSetKernelArg(kernelCGP, 4,sizeof(cl_mem), bufferFunctions);
    clSetKernelArg(kernelCGP, 5,sizeof(cl_mem), bufferSeeds);
    clSetKernelArg(kernelCGP, 6,sizeof(cl_mem), bufferPopulation);
    clSetKernelArg(kernelCGP, 7,sizeof(cl_mem), bufferBest);
    clSetKernelArg(kernelCGP, 8, (int)localSizeAval * sizeof(float), nullptr);

    kernelEvolve = clCreateKernel(program, "evolve", &result);
    checkError(result);
    clSetKernelArg(kernelEvolve, 0, sizeof(cl_mem), bufferFunctions);
    clSetKernelArg(kernelEvolve, 1, sizeof(cl_mem), bufferSeeds);
    clSetKernelArg(kernelEvolve, 2, sizeof(cl_mem), bufferPopulation);
    clSetKernelArg(kernelEvolve, 3, sizeof(cl_mem), bufferBest);

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
    result = clEnqueueWriteBuffer(cmdQueue, bufferSeeds, CL_FALSE, 0, NUM_INDIV * maxLocalSize  * sizeof(int), seeds,
                                  0, nullptr, nullptr);
    checkError(result);

    result = clEnqueueWriteBuffer(cmdQueue, bufferDataOut, CL_FALSE, 0, numPoints * (p->N + p->O) * sizeof(float), transposeDatasetOutput, 0,
                                  nullptr, nullptr);

    result = clEnqueueWriteBuffer(cmdQueue, bufferDatasetTrain, CL_FALSE, 0, numPointsTrain * p->N * sizeof(float), transposeDatasetTrain, 0,
                                  nullptr, nullptr);
    result = clEnqueueWriteBuffer(cmdQueue, bufferOutputsTrain, CL_FALSE, 0, numPointsTrain * p->O * sizeof(float), transposeOutputsTrain, 0,
                                  nullptr, nullptr);

    result = clEnqueueWriteBuffer(cmdQueue, bufferDatasetValid, CL_FALSE, 0, numPointsValid * p->N * sizeof(float), transposeDatasetValid, 0,
                                  nullptr, nullptr);
    result = clEnqueueWriteBuffer(cmdQueue, bufferOutputsValid, CL_FALSE, 0, numPointsValid * p->O * sizeof(float), transposeOutputsValid, 0,
                                  nullptr, nullptr);

    result = clEnqueueWriteBuffer(cmdQueue, bufferDatasetTest, CL_FALSE, 0, numPointsTest * p->N * sizeof(float), transposeDatasetTest, 0,
                                  nullptr, nullptr);
    result = clEnqueueWriteBuffer(cmdQueue, bufferOutputsTest, CL_FALSE, 0, numPointsTest * p->O * sizeof(float), transposeOutputsTest, 0,
                                  nullptr, nullptr);

    result = clEnqueueWriteBuffer(cmdQueue, bufferFunctions, CL_FALSE, 0, (p->NUM_FUNCTIONS) * sizeof(unsigned int), p->functionSet, 0,
                                  nullptr, nullptr);

    clFinish(cmdQueue);
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
    clEnqueueWriteBuffer(cmdQueue, bufferBest, CL_FALSE, 0, sizeof(Chromosome), best, 0, nullptr, nullptr);
}

void OCLConfig::writePopulationBuffer(Chromosome* population){
    clEnqueueWriteBuffer(cmdQueue, bufferPopulation, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), population, 0, nullptr,
                         nullptr);
}

void OCLConfig::readBestBuffer(Chromosome* best){
    clEnqueueReadBuffer(cmdQueue, bufferBest, CL_FALSE, 0,  sizeof(Chromosome), best, 0, nullptr, nullptr);
}

void OCLConfig::readPopulationBuffer(Chromosome* population){
    clEnqueueReadBuffer(cmdQueue, bufferPopulation, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), population, 0, nullptr,
                        nullptr);
}

void OCLConfig::readSeedsBuffer(int* seeds){
    clEnqueueReadBuffer(cmdQueue, bufferSeeds, CL_FALSE, 0, NUM_INDIV * maxLocalSize  * sizeof(int), seeds, 0, nullptr,
                        nullptr);
}

void OCLConfig::finishCommandQueue(){
    int result = clFinish(cmdQueue);
    checkError(result);
}

void OCLConfig::enqueueCGPKernel(){
    int result = clEnqueueNDRangeKernel(cmdQueue, kernelCGP, 1, nullptr, &globalSizeAval, &localSizeAval, 0,
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueTestKernel(){
    int result = clEnqueueNDRangeKernel(cmdQueue, kernelTest, 1, nullptr, &globalSizeTest, &localSizeTest, 0,
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueEvolveKernel(){
    int result = clEnqueueNDRangeKernel(cmdQueue, kernelEvolve, 1, nullptr, &globalSizeEvol, &localSizeEvol, 0,
                                               nullptr, &e_tempo);
    checkError(result);
}

void OCLConfig::enqueueEvaluationKernel(){
    int result = clEnqueueNDRangeKernel(cmdQueue, kernelEvaluate, 1, nullptr, &globalSizeAval, &localSizeAval, 0,
                                               nullptr, &e_tempo);
    checkError(result);
}
/*
void OCLConfig::enqueueEvaluationKernel2(){
    int result = cmdQueue.enqueueNDRangeKernel(kernelEvaluate2, cl::NullRange, cl::NDRange(globalSizeAval), cl::NDRange(localSizeAval),
                                               nullptr, &e_tempo);
    checkError(result);
}
*/
/*
double OCLConfig::getKernelElapsedTime(){
    e_tempo.getProfilingInfo(CL_PROFILING_COMMAND_START, &inicio);
    e_tempo.getProfilingInfo(CL_PROFILING_COMMAND_END, &fim);
    return ((fim-inicio)/1.0E9);
}
 */

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