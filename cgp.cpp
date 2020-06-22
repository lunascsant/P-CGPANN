//
// Created by bruno on 04/02/2020.
//

#include "cgp.h"



void newNode(Chromosome* c, Parameters* params, unsigned int index, int* seed){
    /** set node function */
    c->nodes[index].function = params->functionSet[randomFunction(params, seed)];

    c->nodes[index].maxInputs = getFunctionInputs(c->nodes[index].function);

    /** set node inputs */
    for(int pos = 0; pos < MAX_ARITY; pos++){
        c->nodes[index].inputs[pos] = randomInput(params, index, seed);
        c->nodes[index].inputsWeight[pos] = randomConnectionWeight(params, seed);
    }

    /** set to unactive as default */
    c->nodes[index].active = 0;

    /** set to output 0 as default */
    //c->nodes[index].output = 0;

    //c->nodes[index].inputsEvaluated = 0;

}

void activateNodes(Chromosome* c, Parameters* p){

    int i, j;
    int alreadyEvaluated[MAX_NODES] = {-1};
    for(i = 0; i < MAX_NODES; i++) {
        alreadyEvaluated[i] = -1;
        c->nodes[i].active = 0;
    }
    c->numActiveNodes = 0;

    Stack s;
    s.topIndex = -1;


    for(i = 0; i < MAX_OUTPUTS; i++) {
        unsigned int nodeIndex = c->output[i];
        push(&s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            if( c->nodes[node].active == 0) {
                for (j = 0; j < MAX_ARITY; j++) {
                    if (c->nodes[node].inputs[j] >= p->N) {
                        push(&s, c->nodes[node].inputs[j] - p->N);
                    }
                }

                c->nodes[node].active = 1;
                c->numActiveNodes++;
            }

        }
    }
}

void circuitGenerator(Chromosome* c, Parameters* params, int* seed){
    unsigned int i;

    for(i = 0; i < MAX_NODES; i++){
        newNode(c, params, i, seed);
    }

    for(i = 0; i < MAX_OUTPUTS; i++){
        c->output[i] = randomOutputIndex(seed);
    }

    c->fitness = 0.0;
    c->fitnessValidation = 0.0;
    c->numActiveNodes = 0;
    activateNodes(c, params);
}

/*
void setFitness(Chromosome* c, Parameters* p, float* out, int* fitness) {
    int j;

    float maxPredicted = -DBL_MAX;
    int predictedClass = 0;
    int correctClass = 0;

    for(j = 0; j < MAX_OUTPUTS; j++) {
        float currentPrediction = c->nodes[c->output[j]].output;

        if(currentPrediction > maxPredicted) {
            maxPredicted = currentPrediction;
            predictedClass = j;
        }

        if(out[j] == 1.0) {
            correctClass = j;
        }
    }

    if(predictedClass == correctClass) {
        (*fitness)++;
    }
}
*/

void evaluateCircuit(Chromosome* c, Parameters* p, float** data, float** out) {
    int i;
    int fitness = 0;
    c->fitness = 0.0;
    for(i = 0; i < p->M; i++){
        runCircuit(c, p, data[i], out[i]);
        //setFitness(c, p, out[i], &fitness);
    }
    c->fitness = c->fitness / (float) p->M;
}


float executeFunction(Chromosome* c, int node, ExStack* exStack){
    int i;
    float result, sum;
    unsigned int inputs = c->nodes[node].maxInputs;
    switch (c->nodes[node].function){
        case ADD:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result += exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case ABS:
            result = fabs(popEx(exStack));
            break;
        case SQRT:
            result = sqrt(popEx(exStack));
            break;
        case SQ:
            result = pow(popEx(exStack), 2);
            break;
        case CUBE:
            result = pow(popEx(exStack), 3);
            break;
        case POW:
            result = popEx(exStack);
            result = pow(popEx(exStack), result);
            break;
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 0;
                }
            }
            break;
        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
            break;
        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
            break;
        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
            break;
        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
            break;
        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
            break;
        case NOT:
            result = 0;
            if(popEx(exStack) == 0){
                result = 1;
            }
            break;
        case EXP:
            result = exp(popEx(exStack));
            break;
        case SIN:
            result = sin(popEx(exStack));
            break;
        case COS:
            result = cos(popEx(exStack));
            break;case TAN:
            result = tan(popEx(exStack));
            break;
        case ONE:
            result = 1;
            break;
        case ZERO:
            result = 0;
            break;
        case PI:
            result = CONST_PI;
            break;
        case WIRE:
            result = popEx(exStack);
            break;
        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1 / (1 + exp((-1) * sum));
            break;
        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow(sum - 0, 2)) / (2 * pow(1, 2)));
            break;

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        case RAND:
            //???
        break;
        default:
            break;
    }
    return result;
}


void runCircuit(Chromosome* c, Parameters* p, float* data, float* out){

    int i;
    float maxPredicted = -DBL_MAX;
    int predictedClass = 0;
    int correctClass = 0;

    float executionOut[MAX_OUTPUTS];
    float alreadyEvaluated[MAX_NODES];
    int inputsEvaluatedAux[MAX_NODES];

    for(i = 0; i < MAX_NODES; i++){
        alreadyEvaluated[i] = -FLT_MAX;
        inputsEvaluatedAux[i] = 0;
        //c->nodes[i].inputsEvaluated = 0;
    }

    Stack s;
    s.topIndex = -1;

    ExStack exStack;
    exStack.topIndex = -1;


    for( i = 0; i < MAX_OUTPUTS; i++) {
        unsigned int nodeIndex = c->output[i];
        push(&s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            for (int j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                if (c->nodes[node].inputs[j] >= p->N) { // se é um outro nó, empilha nó ou o resultado
                    unsigned int refIndex = c->nodes[node].inputs[j] - p->N;

                    if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                        inputsEvaluatedAux[node]++;//c->nodes[node].inputsEvaluated++;
                        pushEx(&exStack, alreadyEvaluated[refIndex]);
                    } else {
                        push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                        push(&s, refIndex); //avalia o próximo
                        break;
                    }
                } else {
                    inputsEvaluatedAux[node]++;//c->nodes[node].inputsEvaluated++;
                    pushEx(&exStack, data[c->nodes[node].inputs[j]]);
                }
            }
            if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                //alreadyEvaluated[node] = 1;
            }

        }
        executionOut[i] = alreadyEvaluated[nodeIndex];//c->nodes[c->output[i]].output;//popEx(&exStack);

        if(executionOut[i] > maxPredicted) {
            maxPredicted = executionOut[i];
            predictedClass = i;
        }

        if(out[i] == 1.0) {
            correctClass = i;
        }

    }

    if(predictedClass == correctClass) {
        (c->fitness)++;
    }

}

void initializePopulation(Chromosome* pop, Parameters* p, int* seed) {
    for(int i = 0; i < NUM_INDIV; i++){
        circuitGenerator(&pop[i], p, seed);
    }
}

int evaluatePopulation(Chromosome* pop, Parameters* p, float** data, float** out){
    int i, j;
    float bestFitness = 0;
    unsigned int bestActiveNodes = 999;
    int bestIndex = -1;

    for(j = 0; j < NUM_INDIV; j++) {
        evaluateCircuit(&pop[j], p, data, out);
        if(pop[j].fitness > bestFitness){
            bestFitness = pop[j].fitness;
            bestActiveNodes = pop[j].numActiveNodes;
            bestIndex = j;
        } else if (pop[j].fitness == bestFitness) {
            if(pop[j].numActiveNodes <= bestActiveNodes){
                bestFitness = pop[j].fitness;
                bestActiveNodes = pop[j].numActiveNodes;
                bestIndex = j;
            }
        }
    }
    return bestIndex;
}

Chromosome *mutateTopologyProbabilistic(Chromosome *c, Parameters *p, int *seed, int type) {

    int i, j;
    for(i = 0; i < MAX_NODES; i++){

        if(randomProb(seed) <= PROB_MUT) {
            c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }
        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(p, i, seed);
            }
            if(type == 0 && randomProb(seed) <= PROB_MUT){
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(p, seed);
            }
        }
    }

    activateNodes(c, p);
    return  c;
}

Chromosome *mutateTopologyProbabilisticActive(Chromosome *c, Parameters *p, int *seed, int type) {

    int i, j;
    for(i = 0; i < MAX_NODES; i++){
        if(c->nodes[i].active == 1){
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
                c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
            }
            for(j = 0; j < c->nodes[i].maxInputs; j++) {
                if(randomProb(seed) <= PROB_MUT) {
                    c->nodes[i].inputs[j] = randomInput(p, i, seed);
                }
                if(type == 0 && randomProb(seed) <= PROB_MUT){
                    c->nodes[i].inputsWeight[j] = randomConnectionWeight(p, seed);
                }
            }
        }
    }

    activateNodes(c, p);
    return  c;
}

Chromosome *mutateTopologyPoint(Chromosome *c, Parameters *p, int *seed) {
    int mutationComplete = -1;
    unsigned int newIndex;
    unsigned int newInputIndex;
    unsigned int newValue;

    int num_inputs = MAX_NODES * MAX_ARITY;
    while (mutationComplete == -1){
        unsigned int nodeIndex = randomInterval(0, MAX_NODES + (num_inputs) + MAX_OUTPUTS, seed); //Select any node or output
        if(nodeIndex < MAX_NODES) { // select function
            newIndex = nodeIndex;
            newValue = p->functionSet[randomFunction(p, seed)];
            if(newValue != c->nodes[newIndex].function){
                c->nodes[newIndex].function = newValue;
                c->nodes[newIndex].maxInputs = getFunctionInputs(newValue);
                if(c->nodes[newIndex].active > -1) {
                    mutationComplete = 1;
                }
            }
        } else if (nodeIndex <= MAX_NODES + (num_inputs)) { //select input
            newIndex = (unsigned int) ((nodeIndex - MAX_NODES) / MAX_ARITY);
            newInputIndex= (unsigned int) ((nodeIndex - MAX_NODES) % MAX_ARITY);

            newValue = randomInput(p, newIndex, seed);

            if(newValue != c->nodes[newIndex].inputs[newInputIndex]){
                c->nodes[newIndex].inputs[newInputIndex] = newValue;
                if(c->nodes[newIndex].active == 1) {
                    mutationComplete = 1;
                }
            }

        } else { // select an output
            newIndex = nodeIndex - (MAX_NODES + (num_inputs)) - 1;
            newValue = randomOutputIndex(seed);

            if(newValue != c->output[newIndex]) {
                c->output[newIndex] = newValue;
                mutationComplete = 1;
            }
        }

    }
    activateNodes(c, p);
    return  c;
}


void test(Parameters* p, float** data, float** out){
    Chromosome c;
    c.nodes[0].function  = XOR;
    c.nodes[0].inputs[0] = 1;
    c.nodes[0].inputs[1] = 3;
    //c.nodes[0].inputsEvaluated = 0;
    c.nodes[0].maxInputs = MAX_ARITY;
    c.nodes[1].function  =  AND;
    c.nodes[1].inputs[0] = 1;
    c.nodes[1].inputs[1] = 3;
    //c.nodes[1].inputsEvaluated = 0;
    c.nodes[1].maxInputs = MAX_ARITY;
    c.nodes[2].function  =  XOR;
    c.nodes[2].inputs[0] = 0;
    c.nodes[2].inputs[1] = 2;
    //c.nodes[2].inputsEvaluated = 0;
    c.nodes[2].maxInputs = MAX_ARITY;
    c.nodes[3].function =  AND;
    c.nodes[3].inputs[0] = 0;
    c.nodes[3].inputs[1] = 2;
    //c.nodes[3].inputsEvaluated = 0;
    c.nodes[3].maxInputs = MAX_ARITY;
    c.nodes[4].function  =  XOR;
    c.nodes[4].inputs[0] =  5;
    c.nodes[4].inputs[1] =  6;
    //c.nodes[4].inputsEvaluated = 0;
    c.nodes[4].maxInputs = MAX_ARITY;
    c.nodes[5].function  =  AND;
    c.nodes[5].inputs[0] =  5;
    c.nodes[5].inputs[1] =  6;
    //c.nodes[5].inputsEvaluated = 0;
    c.nodes[5].maxInputs = MAX_ARITY;
    c.nodes[6].function  =  OR;
    c.nodes[6].inputs[0] =  9;
    c.nodes[6].inputs[1] =  7;
    //c.nodes[6].inputsEvaluated = 0;
    c.nodes[6].maxInputs = MAX_ARITY;
    c.output[0] = 6;//0;//0;
    c.output[1] = 4;
    c.output[2] = 0;

    c.fitness = 0.0;
    evaluateCircuit(&c, p, data, out);
    std::cout << "Fitness " << c.fitness << std::endl;


}

Chromosome CGP(Chromosome best, float** dataset, float** outputs, Parameters* params, int *seeds) {
    Chromosome new_best;
    Chromosome mutated_best;

    new_best = best;

    int iterations = 0;
    while(stopCriteria(iterations)) {
        if(iterations % 10 == 0) printf("Generation %d:\n", iterations);
        //printCircuit(&best, params);
        for (int i = 0; i < NUM_INDIV; i++){
            mutated_best = best;
            mutateTopologyProbabilistic(&mutated_best, params, &seeds[i], 0);
            evaluateCircuit(&mutated_best, params, dataset, outputs);
            if(mutated_best.fitness > new_best.fitness){
                new_best = mutated_best;
            } else if (mutated_best.fitness == new_best.fitness) {
                if(mutated_best.numActiveNodes <= new_best.numActiveNodes){
                    new_best = mutated_best;
                }
            }
        }
        best = new_best;
        std::cout << "Best fitness  = " << best.fitness << std::endl;
        iterations++;
    }

    return best;
}


void printOpenclDeviceInfo(std::vector<cl::Platform> platforms, std::vector<cl::Device> devices){

    std::cout << "Available Devices for Platform " << platforms[GPU_PLATFORM].getInfo<CL_PLATFORM_NAME>()<< ":\n";

    for(int i = 0; i < devices.size(); ++i) {
        std::cout << "[" << i << "]" << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "\tType:   " << devices[i].getInfo<CL_DEVICE_TYPE>() << std::endl;
        std::cout << "\tOpenCL: " << devices[i].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        std::cout << "\tMax Comp Un: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        std::cout << "\tMax WrkGr Sz: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        std::cout << "\tFp config: " << devices[i].getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() << std::endl;
        std::cout << "\tMax Mem Alloc: " << devices[i].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
        std::cout << "\tLocal Mem Size: " << devices[i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
        std::cout << "\tMax Const Size: " << devices[i].getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << std::endl;

    }

}

void checkError(cl_int result){
    if(result != CL_SUCCESS)
        std::cerr << getErrorString(result) << std::endl;
}

Chromosome PCGP(Chromosome best,  Dataset* data, Parameters* params, int *seeds){
    ///transposicao de dados
    float* transposeDataset;
    float* transposeOutputs;

    transposeData(data, &transposeDataset, &transposeOutputs);

    Chromosome new_best;
    Chromosome newBest[NUM_INDIV];

    ///OpenCL
    new_best = best;
    cl_int result;

    ///Platforms and Devices
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    setupOpenCLOnePlatform(platforms,devices);
    printOpenclDeviceInfo(platforms, devices);

    ///Context
    cl::Context context(devices, nullptr, nullptr, nullptr, &result);

    ///Command Queue
    cl_command_queue_properties commandQueueProperties = CL_QUEUE_PROFILING_ENABLE;
    cl::CommandQueue cmdQueue(context, devices[GPU_DEVICE], commandQueueProperties, &result);

    ///Buffers
    cl::Buffer bufferSeeds     (context, CL_MEM_READ_WRITE, NUM_INDIV * sizeof(int), nullptr,  &result);
    checkError(result);
    cl::Buffer bufferDataset   (context, CL_MEM_READ_ONLY, data->M * data->N * sizeof(float), nullptr,  &result);
    checkError(result);
    cl::Buffer bufferOutputs   (context, CL_MEM_READ_ONLY, data->M * data->O * sizeof(float), nullptr,  &result);
    checkError(result);
    cl::Buffer bufferFunctions (context, CL_MEM_READ_ONLY, ((params->NUM_FUNCTIONS)) * sizeof(unsigned int), nullptr,  &result);
    checkError(result);
    cl::Buffer bufferBest      (context, CL_MEM_READ_ONLY, sizeof(Chromosome), nullptr,  &result);
    checkError(result);
    cl::Buffer bufferNewBest   (context, CL_MEM_READ_WRITE, NUM_INDIV * sizeof(Chromosome), nullptr,  &result);
    checkError(result);


    ///ND-Ranges
    std::string compileFlags;

    size_t numPoints = (size_t) params->M;;
    size_t maxLocalSize = cmdQueue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    size_t localSize_evol = 1;
    size_t globalSize_evol = NUM_INDIV;
    size_t localSize_aval, globalSize_aval;

    setNDRanges(&globalSize_aval, &localSize_aval, &compileFlags, maxLocalSize, numPoints, cmdQueue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_TYPE>()); /// ok


    ///Program build
    std::ifstream sourceFileName("kernels\\kernel.cl");
    std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName),(std::istreambuf_iterator<char>()));
    std::string program_src = setProgramSource(data, params, (int)localSize_aval) + sourceFile;

    cl::Program program(context, program_src);

    result = program.build(devices, compileFlags.c_str());
    if(result != CL_SUCCESS){
        std::cerr << getErrorString(result) << std::endl;

        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        std::cerr << "Build log:" << std::endl
                  << buildLog << std::endl;
    }


    ///Kernels
    cl::Kernel krnl_evolve(program, "evolve", &result);
    checkError(result);

    cl::Kernel krnl_evaluate(program, "evaluate", &result);
    checkError(result);

    cl::Kernel krnl_cgp(program, "CGP", &result);
    checkError(result);

    ///Write buffers that won't be read back to host
    cmdQueue.enqueueWriteBuffer(bufferSeeds, CL_FALSE, 0, NUM_INDIV * sizeof(int), seeds);
    cmdQueue.enqueueWriteBuffer(bufferDataset, CL_FALSE, 0, params->M * params->N * sizeof(float), transposeDataset);
    cmdQueue.enqueueWriteBuffer(bufferOutputs, CL_FALSE, 0, params->M * params->O * sizeof(float), transposeOutputs);
    cmdQueue.enqueueWriteBuffer(bufferFunctions, CL_FALSE, 0, (params->NUM_FUNCTIONS) * sizeof(unsigned int), params->functionSet);
    std::cout << params->M * params->N * sizeof(float) << std::endl;
    std::cout << params->M * params->O * sizeof(float) << std::endl;

    cmdQueue.finish();



    int iterations = 0;
    while(stopCriteria(iterations)) {
        printf("Generation %d:\n", iterations);

        std::cout << "Active nodes: " << best.numActiveNodes<< ", Fitness: " << best.fitness  << std::endl;

        //timeManager.getStartTime(Iteracao_T);
        ///CGP Kernel

        cmdQueue.enqueueWriteBuffer(bufferBest, CL_FALSE, 0, sizeof(Chromosome), &best);
        cmdQueue.enqueueWriteBuffer(bufferNewBest, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), newBest);

        krnl_cgp.setArg(0, bufferNewBest);
        krnl_cgp.setArg(1, bufferBest);
        krnl_cgp.setArg(2, bufferDataset);
        krnl_cgp.setArg(3, bufferOutputs);
        krnl_cgp.setArg(4, bufferFunctions);
        krnl_cgp.setArg(5, bufferSeeds);
        krnl_cgp.setArg(6, (int)localSize_aval * sizeof(float), nullptr);

        result = cmdQueue.enqueueNDRangeKernel(krnl_cgp, cl::NullRange, cl::NDRange(globalSize_aval), cl::NDRange(localSize_aval));
        checkError(result);

        cmdQueue.finish();

        cmdQueue.enqueueReadBuffer(bufferNewBest, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), newBest);

        cmdQueue.finish();


/*
        ///Evolution Kernel
        cmdQueue.enqueueWriteBuffer(bufferBest, CL_FALSE, 0, sizeof(Chromosome), &best);
        cmdQueue.enqueueWriteBuffer(bufferNewBest, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), newBest);

        krnl_evolve.setArg(0, bufferBest);
        krnl_evolve.setArg(1, bufferNewBest);
        krnl_evolve.setArg(2, bufferFunctions);
        krnl_evolve.setArg(3, bufferSeeds);
        result = cmdQueue.enqueueNDRangeKernel(krnl_evolve, cl::NullRange, cl::NDRange(globalSize_evol), cl::NDRange(localSize_evol));
        checkError(result);
        cmdQueue.finish();


        ///Evaluation Kernel
        krnl_evaluate.setArg(0, bufferNewBest);
        krnl_evaluate.setArg(1, bufferDataset);
        krnl_evaluate.setArg(2, bufferOutputs);
        krnl_evaluate.setArg(3, bufferFunctions);
        krnl_evaluate.setArg(4, (int)localSize_aval * sizeof(float), nullptr);
        result = cmdQueue.enqueueNDRangeKernel(krnl_evaluate, cl::NullRange, cl::NDRange(globalSize_aval), cl::NDRange(localSize_aval));
        checkError(result);
        cmdQueue.finish();

        cmdQueue.enqueueReadBuffer(bufferNewBest, CL_FALSE, 0, NUM_INDIV * sizeof(Chromosome), newBest);
        cmdQueue.finish();
*/
        for(int k = 0; k < NUM_INDIV; k++){
            if(newBest[k].fitness > best.fitness){
                best = newBest[k];
            } else if (newBest[k].fitness == best.fitness) {
                if(newBest[k].numActiveNodes <= best.numActiveNodes){
                    best = newBest[k];
                }
            }
        }
        iterations++;

        //timeManager.getEndTime(Iteracao_T);
        //timeManager.getElapsedTime(Iteracao_T);
    }
    return best;
}



Chromosome CGPDE_IN();

Chromosome CGPDE_OUT();

Chromosome PCGPDE_IN();

Chromosome PCGPDE_OUT();
