//
// Created by bruno on 04/02/2020.
//

#include "utils.h"

int rand2(int *seed){
    int s  = *seed;
    s = ((unsigned int)(s * 16807) % 2147483647);//(int)(pown(2.0, 31)-1));
    *seed = s;

    return s;
}


unsigned int randomInput(Parameters *p, unsigned int index, int *seed) {
    return (rand2(seed) % (p->N + index));
}

unsigned int randomOutputIndex(int* seed){
    return (rand2(seed) % MAX_NODES);
}

unsigned int randomFunction(Parameters *p, int *seed) {
    return (rand2(seed) % (p->NUM_FUNCTIONS));
}

float randomConnectionWeight(Parameters *p, int *seed) {
    return (((float) rand2(seed) / (float) (2147483647) ) * 2 * p->weightRange) - p->weightRange;
}

int randomInterval(int inf_bound, int sup_bound, int *seed) {
    return rand2(seed) % (sup_bound - inf_bound + 1) + inf_bound;
}

float randomProb(int* seed){
    return (float)rand2(seed) / 2147483647;//pown(2.0, 31);
}

unsigned int getFunctionInputs(unsigned int function){
    switch (function) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case AND:
        case OR:
        case XOR:
        case NAND:
        case NOR:
        case XNOR:
        case SIG:
        case GAUSS:
        case STEP:
        case SOFTSIGN:
        case TANH:
            return MAX_ARITY;
        case RAND:
        case PI:
        case ONE:
        case ZERO:
            return 0;
        case ABS:
        case SQRT:
        case SQ:
        case CUBE:
        case EXP:
        case SIN:
        case COS:
        case TAN:
        case NOT:
        case WIRE:
            return 1;
        case POW:
            return 2;
        default:
            break;
    }
}

bool IsPowerOf2( int n ){
    return (n & -n) == n;
}

unsigned NextPowerOf2( unsigned n ){
    n--;
    n |= n >> 1;  // handle  2 bit numbers
    n |= n >> 2;  // handle  4 bit numbers
    n |= n >> 4;  // handle  8 bit numbers
    n |= n >> 8;  // handle 16 bit numbers
    n |= n >> 16; // handle 32 bit numbers
    n++;

    return n;
}
void readDataset(Parameters* params, Dataset* fulldata, char* filename){

    std::fstream arq;

    int i, j, k;
    int readLabel = 0;
    int readOps;
    int info;

    printf("Lendo Dados Arquivo... %s\n",filename);
    arq.open(filename, std::fstream::in);

    /** Read the dataset size (M) and number of inputs (N) */
    std::string value;
/*
    arq >> value;
    if(value == ".p")
        arq >> (params->M);
    arq >> value;
    if(value == ".i")
        arq >> (params->N);
    arq >> value;
    if(value == ".o")
        arq >> (params->O);
*/
    arq >> (params->N);
    arq >> (params->O);
    arq >> (params->M);


    //arq >> (readLabel);

    unsigned int M = params->M;
    unsigned int N = params->N;
    unsigned int O = params->O;
    //std::cout << M << " " << N << " " << O << std::endl;

    fulldata->M = M;
    fulldata->N = N;
    fulldata->O = O;

    (fulldata->data) = new float* [(M)];
    for(i = 0; i < (M); i++){
        (fulldata->data)[i] = new float [(N)];
    }

    (fulldata->output) = new float* [(M)];
    for(i = 0; i < (M); i++) {
        (fulldata->output)[i] = new float[(O)];
    }

    (params->labels) = new char* [(N + O)];
    for(i = 0; i < (N + O); i++){
        (params->labels)[i] = new char [10];
    }

    //LABELS
    for(i = 0; i < params->N; i++){
        std::stringstream ss;
        std::string str;
        ss << "i";
        ss << i;
        ss >> str;
        strcpy((params->labels)[i], (str.c_str()));
    }
    for(; i < params->N+params->O; i++){
        std::stringstream ss;
        std::string str;
        ss << "o";
        ss << i;
        ss >> str;
        strcpy((params->labels)[i], (str.c_str()));
    }


    /** Read the dataset */
    std::string line;
    for(i = 0; i < (M); i++){
        //arq >> line;
        //std::cout << line <<std::endl;
        for(j = 0; j < (N); j++){
            arq >> (fulldata->data)[i][j] ;//= line[j] - '0';
            //std::cout << (*dataset)[i][j] << " ";
        }
        for(k = 0; j<(N+O); j++, k++){
            arq >> (fulldata->output)[i][k];// = line[j] - '0';
            //std::cout << (*outputs)[i][k] << " ";
        }
        //std::cout << std::endl;
    }

    arq >> readOps;


    params->NUM_FUNCTIONS = 1;
    (params->functionSet) = new unsigned int [params->NUM_FUNCTIONS];

    i = 0;

    (params->functionSet)[i++] = SIG;
    //(params->maxFunctionInputs)[i++] = 2;
/*
    (params->functionSet)[i++] = OR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = XOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = NAND;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = NOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = XNOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = NOT;
    //(params->maxFunctionInputs)[i++] = 1;
*/


    params->weightRange = 5;
}

void readDataset(Parameters* params, float*** dataset, float*** outputs, char* filename){

    std::fstream arq;

    int i, j, k;
    int readLabel = 0;
    int readOps;
    int info;

    printf("Lendo Dados Arquivo... %s\n",filename);
    arq.open(filename, std::fstream::in);

    /** Read the dataset size (M) and number of inputs (N) */
    std::string value;
/*
    arq >> value;
    if(value == ".p")
        arq >> (params->M);
    arq >> value;
    if(value == ".i")
        arq >> (params->N);
    arq >> value;
    if(value == ".o")
        arq >> (params->O);
*/
    arq >> (params->N);
    arq >> (params->O);
    arq >> (params->M);


    //arq >> (readLabel);

    unsigned int M = params->M;
    unsigned int N = params->N;
    unsigned int O = params->O;
    //std::cout << M << " " << N << " " << O << std::endl;

    (*dataset) = new float* [(M)];
    for(i = 0; i < (M); i++){
        (*dataset)[i] = new float [(N)];
    }

    (*outputs) = new float* [(M)];
    for(i = 0; i < (M); i++) {
        (*outputs)[i] = new float[(O)];
    }

    (params->labels) = new char* [(N + O)];
    for(i = 0; i < (N + O); i++){
        (params->labels)[i] = new char [10];
    }

    //LABELS
    for(i = 0; i < params->N; i++){
        std::stringstream ss;
        std::string str;
        ss << "i";
        ss << i;
        ss >> str;
        strcpy((params->labels)[i], (str.c_str()));
    }
    for(; i < params->N+params->O; i++){
        std::stringstream ss;
        std::string str;
        ss << "o";
        ss << i;
        ss >> str;
        strcpy((params->labels)[i], (str.c_str()));
    }


    /** Read the dataset */
    std::string line;
    for(i = 0; i < (M); i++){
        //arq >> line;
        //std::cout << line <<std::endl;
        for(j = 0; j < (N); j++){
            arq >> (*dataset)[i][j] ;//= line[j] - '0';
            //std::cout << (*dataset)[i][j] << " ";
        }
        for(k = 0; j<(N+O); j++, k++){
            arq >> (*outputs)[i][k];// = line[j] - '0';
            //std::cout << (*outputs)[i][k] << " ";
        }
        //std::cout << std::endl;
    }

    arq >> readOps;


    params->NUM_FUNCTIONS = 1;
    (params->functionSet) = new unsigned int [params->NUM_FUNCTIONS];

    i = 0;

    (params->functionSet)[i++] = SIG;
    //(params->maxFunctionInputs)[i++] = 2;
/*
    (params->functionSet)[i++] = OR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = XOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = NAND;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = NOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = XNOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[i++] = NOT;
    //(params->maxFunctionInputs)[i++] = 1;
*/


    params->weightRange = 5;
}

void printDataset(Parameters *params, float **dataset, float **outputs){
    unsigned int i, j;

    for(j = 0; j < params->N; j++) {
        std::cout << params->labels[j] << " ";
    }
    std::cout << "| ";
    for(; j < params->O + params->N; j++) {
        std::cout <<  params->labels[j] << " ";
    }
    std::cout << std::endl;


    for(i = 0; i < params->M; i++){
        for(j = 0; j < params->N; j++) {
            std::cout << dataset[i][j] << " ";
        }
        std::cout << "| ";
        for(j = 0; j < params->O; j++) {
            std::cout << outputs[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

bool stopCriteria(unsigned int it){
    return it < NUM_GENERATIONS;
    //return (it * NUM_INDIV < NUM_EVALUATIONS);
}


const char *getErrorString(cl_int error) {
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


/**OpenCL**/
void setupOpenCLOnePlatform(std::vector<cl::Platform> &platforms, std::vector<cl::Device> &devices){
    ///Encontrando as plataformas disponiveis
    int result = cl::Platform::get(&platforms);
    if(result != CL_SUCCESS){
        std::cout << "Erro ao encontrar plataformas." << std::endl;
        exit(1);
    }
    std::cout << "Available platforms: " << std::endl;

    for(int i = 0; i < platforms.size(); i++){
        std::cout << "\t" << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    ///Encontrando os dispositivos disponiveis na plataforma.
#if GPU
    platforms[GPU_PLATFORM].getDevices(CL_DEVICE_TYPE_ALL, &devices);
#else
    platforms[CPU_PLATFORM].getDevices(CL_DEVICE_TYPE_ALL, &devices);
#endif

    std::cout << std::endl;
}

std::string ToString( float t ){
    std::stringstream ss; ss << std::setprecision(32) << t; return ss.str();
}

std::string setProgramSource(Dataset* data, Parameters* p, int localSize){
    std::string program_src =
            "#define SEED "  + ToString( SEED ) + "\n" +
            "#define N   " + ToString( data->N ) + "\n" +
            "#define O   " + ToString( data->O ) + "\n" +
            "#define M   " + ToString( data->M ) + "\n" +
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
            "#define LOCAL_SIZE " + ToString( localSize ) + "\n";
    return program_src;
}//96468

void setNDRanges(size_t* globalSize, size_t* localSize, std::string* compileFlags, size_t maxLocalSize, size_t numPoints, cl_device_type deviceType){
    //FOR GPU
    /*if(deviceType == CL_DEVICE_TYPE_GPU){*/
    std::cout << "Definindo NDRanges para avaliacao ...";
    if(numPoints < maxLocalSize){
        *localSize = numPoints;
    }else{
        *localSize = maxLocalSize;
    }

    // One individual per work-group
    *globalSize = (*localSize) * NUM_INDIV;

    //if( MAX_NOS > (*localSize) )
    //    (*compileFlags) += " -D PROGRAM_TREE_DOES_NOT_FIT_IN_LOCAL_SIZE";

    if( !IsPowerOf2( *localSize ) )
        (*compileFlags) += " -D LOCAL_SIZE_IS_NOT_POWER_OF_2";

    if( numPoints % (*localSize) != 0 )
        (*compileFlags) += " -D NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE";

    ///FOR CPU
    /*  } else if (deviceType == CL_DEVICE_TYPE_CPU){
          std::cout << "Definindo NDRanges para avaliacao em CPU..." << std::endl;
          *localSize = 1;//m_num_points;
          *globalSize = NUM_INDIV;
      }*/

    (*compileFlags) += " -D LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2="
                       + ToString( NextPowerOf2(*localSize) );

    std::cout << "...fim." << std::endl;

    std::cout << "Global Size = " << globalSize << std::endl << "Local size = " << localSize << std::endl << std::endl;
}


Dataset* generateFolds(Dataset* data){
    int i, j, k, l, count;
    Dataset* folds;
    folds = new Dataset[10];

    for (i = 0; i < 10; i++)
    {
        folds[i].N = data->N;
        folds[i].O = data->O;
        folds[i].M = 0;
    }

    i = 0;
    count = 0;
    while(1) // set the size of each fold
    {
        folds[i].M = folds[i].M + 1;
        count++;
        if(count == data->M)
            break;
        if(i == 9)
            i = 0;
        else
            i++;
    }

    // allocate memory for the folds data
    for(i = 0; i < 10; i++) // for each fold
    {
        folds[i].data = new float* [folds[i].M];
        folds[i].output = new float* [folds[i].M];

        for(j = 0; j < folds[i].M; j++) // for each instance of each fold
        {
            folds[i].data[j] = new float [folds[i].N];
            folds[i].output[j] = new float [folds[i].O];
        }
    }


    // keep the same class proportion in each fold
    int counter[10];// k = 10 // = (int*)malloc(10*sizeof(int));
    for(i = 0; i < 10; i++)
    {
        counter[i] = 0;
    }

    k = 0;
    for(i = 0; i < data->O; i++) // for each class
    {
        for(j = 0; j < data->M; j++) // for each instance
        {
            if(data->output[j][i] == 1.0)
            {
                for (l = 0; l < data->N; l++)
                {
                    folds[k].data[counter[k]][l] = data->data[j][l];
                }

                for (l = 0; l < data->O; l++)
                {
                    folds[k].output[counter[k]][l] = data->output[j][l];
                }

                counter[k] = counter[k] + 1;
                if(k == 9)
                    k = 0;
                else
                    k++;
            }
        }
    }

    return folds;
}

Dataset* generateFolds(float** dataset, float** outputs, Parameters* p){
    int i, j, k, l, count;
    Dataset* folds;
    folds = new Dataset[10];

    for (i = 0; i < 10; i++)
    {
        folds[i].N = p->N;
        folds[i].O = p->O;
        folds[i].M = 0;
    }

    i = 0;
    count = 0;
    while(1) // set the size of each fold
    {
        folds[i].M = folds[i].M + 1;
        count++;
        if(count == p->M)
            break;
        if(i == 9)
            i = 0;
        else
            i++;
    }

    // allocate memory for the folds data
    for(i = 0; i < 10; i++) // for each fold
    {
        folds[i].data = new float* [folds[i].M];
        folds[i].output = new float* [folds[i].M];

        for(j = 0; j < folds[i].M; j++) // for each instance of each fold
        {
            folds[i].data[j] = new float [folds[i].N];
            folds[i].output[j] = new float [folds[i].O];
        }
    }


    // keep the same class proportion in each fold
    int counter[10];// k = 10 // = (int*)malloc(10*sizeof(int));
    for(i = 0; i < 10; i++)
    {
        counter[i] = 0;
    }

    k = 0;
    for(i = 0; i < p->O; i++) // for each class
    {
        for(j = 0; j < p->M; j++) // for each instance
        {
            if(outputs[j][i] == 1.0)
            {
                for (l = 0; l < p->N; l++)
                {
                    folds[k].data[counter[k]][l] = dataset[j][l];
                }

                for (l = 0; l < p->O; l++)
                {
                    folds[k].output[counter[k]][l] = outputs[j][l];
                }

                counter[k] = counter[k] + 1;
                if(k == 9)
                    k = 0;
                else
                    k++;
            }
        }
    }

    return folds;
}

void transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs){

    (*transposeDataset) = new float [data->M * data->N];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    unsigned int pos = 0;
    std::cout << "Transpondo dados..." << std::endl;
    for(int j = 0; j < data->N; ++j ){
        for(int i = 0; i < data->M; ++i ){
            (*transposeDataset)[pos++] = data->data[i][j];
        }
    }

    (*transposeOutputs) = new float [data->M * data->O];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    pos = 0;
    std::cout << "Transpondo outputs..." << std::endl;
    for(int j = 0; j < data->O; ++j ){
        for(int i = 0; i < data->M; ++i ){
            (*transposeOutputs)[pos++] = data->output[i][j];
        }
    }

}