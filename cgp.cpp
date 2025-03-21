//
// Created by bruno on 04/02/2020.
//

#include "cgp.h"
#include "GPTime.h"

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

/*
	used by qsort in sortIntArray
*/
static int cmpInt(const void * a, const void * b) {
    return ( *(int*)a - * (int*)b );
}

void sortActiveArray(unsigned int *array, const int length) {

    qsort(array, length, sizeof(int), cmpInt);
}



void activateNodes(Chromosome* c, Parameters* p){

    int i, j;
    int alreadyEvaluated[MAX_NODES];
    for(i = 0; i < MAX_NODES; i++) {
        alreadyEvaluated[i] = -1;
        c->activeNodes[i] = MAX_NODES + 1;
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
                c->activeNodes[c->numActiveNodes] = node;
                c->nodes[node].active = 1;
                c->numActiveNodes++;
            }

        }
    }
    sortActiveArray(c->activeNodes, c->numActiveNodes);
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

void initializePopulation(Chromosome* pop, Parameters* p, int* seed) {
    for(int i = 0; i < NUM_INDIV; i++){
        circuitGenerator(&pop[i], p, seed);
    }
}


void evaluateCircuit(Chromosome* c, Dataset* data) {
    int i;
    c->fitness = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuit(c, data, i, 0);
        //setFitness(c, p, out[i], &fitness);
    }

    //std::cout << "FITNESS INDIVIDUO " << c->fitness << std::endl;
    // c->fitness = c->fitness / (float) data->M;
}

void evaluateCircuitValidation(Chromosome* c, Dataset* data) {
    int i;
    c->fitnessValidation = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuit(c, data, i, 1);
        //setFitness(c, p, out[i], &fitness);
    }
    c->fitnessValidation = c->fitnessValidation / (float) data->M;
}

void evaluateCircuitLinear(Chromosome* c, Dataset* data) {
    int i;
    c->fitness = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuitLinear(c, data, i, 0);
        //setFitness(c, p, out[i], &fitness);
    }
    c->fitness = c->fitness / (float) data->M;
}

void evaluateCircuitValidationLinear(Chromosome* c, Dataset* data) {
    int i;
    c->fitnessValidation = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuitLinear(c, data, i, 1);
        //setFitness(c, p, out[i], &fitness);
    }
    c->fitnessValidation = c->fitnessValidation / (float) data->M;
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
            result = 1.0f / (1 + exp(-sum));
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

void runCircuit(Chromosome* c, Dataset* dataset, int index, int validation){

    /*std::cout << "RODANDO UM INDIVIDUO" << std::endl;*/
    int i;
    float maxPredicted = -DBL_MAX;
    int predictedClass = 0;
    int correctClass = 0;

    float executionOut[MAX_OUTPUTS];
    float alreadyEvaluated[MAX_NODES];
    int inputsEvaluatedAux[MAX_NODES];

    for(i = 0; i < MAX_NODES; i++){
        alreadyEvaluated[i] = -DBL_MAX;
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
                if (c->nodes[node].inputs[j] >= dataset->N) { // se é um outro nó, empilha nó ou o resultado
                    unsigned int refIndex = c->nodes[node].inputs[j] - dataset->N;

                    if(alreadyEvaluated[refIndex] > -DBL_MAX) {
                        inputsEvaluatedAux[node]++;//c->nodes[node].inputsEvaluated++;
                        //c->nodes->inputsOutputs[j] = alreadyEvaluated[refIndex];
                        pushEx(&exStack, alreadyEvaluated[refIndex]);
                    } else {
                        push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                        push(&s, refIndex); //avalia o próximo
                        break;
                    }
                } else {
                    inputsEvaluatedAux[node]++;//c->nodes[node].inputsEvaluated++;
                    //c->nodes->inputsOutputs[j] = dataset->data[index][c->nodes[node].inputs[j]];
                    pushEx(&exStack, dataset->data[index][c->nodes[node].inputs[j]]);
                }
            }

            if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                if(!(alreadyEvaluated[node] > -DBL_MAX)) {
                    alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                }
                //alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                //pushEx(&exStack, alreadyEvaluated[node]);

                //alreadyEvaluated[node] = 1;
            }

        }
        executionOut[i] = alreadyEvaluated[nodeIndex];//c->nodes[c->output[i]].output;//popEx(&exStack);
        /*std::cout << "---------------------------" << std::endl;
        std::cout << "executionOut[i] " << executionOut[i] << std::endl;
        std::cout << "maxPredicted " << maxPredicted << std::endl;
        std::cout << "output[i] " << c->output[i] << std::endl;
        std::cout << "dataset " << dataset->output[index][i] << std::endl;
        std::cout << "---------------------------" << std::endl;*/

        /*if(executionOut[i] > maxPredicted) {
            maxPredicted = executionOut[i];
            predictedClass = i;
        }

        if(dataset->output[index][i] == 1.0) {
            correctClass = i;
        }*/

        if(dataset->output[index][i] == executionOut[i]) {
            (c->fitness)++;
        }

    }

    /*std::cout << "Fitness " << c->fitness << std::endl;*/

    /*if(predictedClass == correctClass) {
        if(validation == 1){
            (c->fitnessValidation)++;
        } else {
            (c->fitness)++;
        }

    }*/
    /*std::cout << "RODANDO UM INDIVIDUO" << std::endl;*/
}

void runCircuitLinear(Chromosome* c, Dataset* dataset, int index, int validation){

    int i, j, currentActive, activeInputs;
    float maxPredicted = -DBL_MAX;
    int predictedClass = 0;
    int correctClass = 0;

    float executionOut[MAX_OUTPUTS];
    float alreadyEvaluated[MAX_NODES];

    for(i = 0; i < MAX_NODES; i++){
        alreadyEvaluated[i] = -DBL_MAX;
    }

    ExStack exStack;
    exStack.topIndex = -1;

    for(i = 0; i < c->numActiveNodes; i++){
        currentActive = c->activeNodes[i];
        activeInputs = c->nodes[currentActive].maxInputs;

        for(j = 0; j < activeInputs; j++){
            if (c->nodes[currentActive].inputs[j] >= dataset->N) { // se é um outro nó, empilha nó ou o resultado
                unsigned int refIndex = c->nodes[currentActive].inputs[j] - dataset->N;

                if(alreadyEvaluated[refIndex] > -DBL_MAX) {
                    pushEx(&exStack, alreadyEvaluated[refIndex]);
                } else {
                    printf("ERRO. \n");
                }
            } else {
                pushEx(&exStack, dataset->data[index][c->nodes[currentActive].inputs[j]]);
            }
        }

        alreadyEvaluated[currentActive] = executeFunction(c, currentActive, &exStack);
/*

        if (std::isnan(alreadyEvaluated[currentActive]) != 0) {
            alreadyEvaluated[currentActive] = 0;
        }

        else if (std::isinf(alreadyEvaluated[currentActive]) != 0 ) {

            if (alreadyEvaluated[currentActive] > 0) {
                alreadyEvaluated[currentActive] = DBL_MAX;
            }
            else {
                alreadyEvaluated[currentActive] = DBL_MIN;
            }
        }
*/
    }

    for( i = 0; i < MAX_OUTPUTS; i++) {
        unsigned int nodeIndex = c->output[i];

        executionOut[i] = alreadyEvaluated[nodeIndex];//c->nodes[c->output[i]].output;//popEx(&exStack);

        if(executionOut[i] > maxPredicted) {
            maxPredicted = executionOut[i];
            predictedClass = i;
        } else {
            maxPredicted = maxPredicted;
            predictedClass = predictedClass;
        }

        correctClass = (dataset->output[index][i] == 1.0) ? i : correctClass;

    }

    if(validation == 1){
        c->fitnessValidation = (predictedClass == correctClass) ? c->fitnessValidation+1 : c->fitnessValidation+0;
    } else {
        c->fitness = (predictedClass == correctClass) ? c->fitness+1 : c->fitness+0;
    }
}

int evaluatePopulation(Chromosome* pop, Dataset* dataset, int validation){
    int i, j;
    float bestFitness = 0;
    unsigned int bestActiveNodes = 999;
    int bestIndex = -1;
    if(validation == 1 ){
        for(j = 0; j < NUM_INDIV; j++) {
            evaluateCircuitValidation(&pop[j], dataset);
            if(pop[j].fitnessValidation > bestFitness){
                bestFitness = pop[j].fitnessValidation;
                bestActiveNodes = pop[j].numActiveNodes;
                bestIndex = j;
            } else if (pop[j].fitnessValidation == bestFitness) {
                if(pop[j].numActiveNodes <= bestActiveNodes){
                    bestFitness = pop[j].fitnessValidation;
                    bestActiveNodes = pop[j].numActiveNodes;
                    bestIndex = j;
                }
            }
        }
    } else {
        for(j = 0; j < NUM_INDIV; j++) {
            evaluateCircuit(&pop[j], dataset);
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

void copyNode(Node* n, ActiveNode* an){
    an->function = n->function;
    an->maxInputs = n->maxInputs;
    for(int i = 0; i < MAX_ARITY; i++){
        an->inputs[i] = n->inputs[i];
        an->inputsWeight[i] = n->inputsWeight[i];
    }
}

void copyActiveNodes(Chromosome *c, ActiveChromosome* ac){

    for(int i = 0; i < NUM_INDIV; i++ ){

        unsigned int numActiveNodes = c[i].numActiveNodes;
        ac[i].numActiveNodes = numActiveNodes;


        for(int j = 0; j < numActiveNodes; j++){
            unsigned int currentActive = c[i].activeNodes[j];

            copyNode(&(c[i].nodes[currentActive]), &(ac[i].nodes[j]));
            ac[i].nodes[j].originalIndex = currentActive;
        }


        for(int j = 0; j < MAX_OUTPUTS; j++){
            ac[i].output[j] = c[i].output[j];
        }


    }
}

Chromosome *mutateTopologyProbabilistic2(Chromosome *c, Parameters *p, int *seeds, int type, int index) {

    int i, j;

    for(i = 0; i < MAX_NODES; i++){
        int nodeIndex  = index * 1024 + ((i*MAX_ARITY) % 1024);
        //std::cout << "morreu aqui?  " << i << std::endl;
        //std::cout << "morreu aqui pq?  " << nodeIndex << std::endl;
        /*if(randomProb(&seeds[nodeIndex]) <= PROB_MUT) {
            c->nodes[i].function = p->functionSet[randomFunction(p, &seeds[nodeIndex])];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }
*/
        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            int arrayIndex = index * 1024 + ((i*MAX_ARITY + j) % 1024);
            if(randomProb(&seeds[arrayIndex]) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(p, i, &seeds[arrayIndex]);
            }
            /*if(type == 0 && randomProb(&seeds[arrayIndex]) <= PROB_MUT){
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(p, &seeds[arrayIndex]);
            }*/
            if(j == 0) {
                if(randomProb(&seeds[arrayIndex]) <= PROB_MUT) {
                    c->nodes[i].function = p->functionSet[randomFunction(p, &seeds[arrayIndex])];
                    c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
                }
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
                //std::cout << "ANTES " << c->nodes[i].function << std::endl;
                c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
                //c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
                //std::cout << "DEPOIS " << c->nodes[i].function << std::endl;
            }
            for(j = 0; j < c->nodes[i].maxInputs; j++) {
                if(randomProb(seed) <= PROB_MUT) {
                    //std::cout << "ANTES " << c->nodes[i].inputs[j] << std::endl;
                    c->nodes[i].inputs[j] = randomInput(p, i, seed);
                    //std::cout << "DEPOIS " << c->nodes[i].inputs[j] << std::endl;
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

Chromosome *mutateSAM(Chromosome *c, Parameters *p, int *seed) {

    int i, j, inputOrFunction, nodeOrOutput;
    int activeSelected = 0;

    nodeOrOutput = randomInterval(0, MAX_NODES + p->O - 1, seed);

    if (nodeOrOutput > MAX_NODES - 1) {
        activeSelected = 1;
        c->output[0] = randomInterval(0, MAX_NODES - 1, seed);
    } else {
        while(activeSelected == 0) {
            i = randomInterval(0, MAX_NODES - 1, seed);
            if(c->nodes[i].active == 1) {
                activeSelected = 1;
                inputOrFunction = randomInterval(0, 1, seed);
                if (!inputOrFunction) {
                    c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
                } else {
                    j = randomInterval(0, 1, seed);
                    c->nodes[i].inputs[j] = randomInput(p, i, seed);
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




Chromosome
CGP(Dataset *training, Parameters *params, int *seeds, double *timeIter, double *timeKernel, std::ofstream& factivel_file) {
    GPTime timeManager(4);
    Chromosome *current_pop;
    current_pop = new Chromosome[NUM_INDIV];

    int factivel = 0;

    Chromosome best;
    Chromosome best_train;
    //Chromosome best_valid;
    Chromosome mutated_best;

    initializePopulation(current_pop, params, &seeds[0]);

    int bestTrain = evaluatePopulation(current_pop, training, 0);
    //int bestValid = evaluatePopulation(current_pop, validation, 1);


    best_train = current_pop[bestTrain];
    //best_valid = current_pop[bestValid];
    best = best_train;

    if(best.fitness == training->M) {
        //std::cout << "CGP achou o indivíduo factivel" << std::endl;
        //printChromosome(&best, params);
        printFile(&best, params, factivel_file);
        factivel = 1;
    }

    //std::cout << "Melhor da populacao: " << best.fitness << std::endl;

    int iterations = 0;
    while(stopCriteria(iterations) && (factivel != 1)) {
        timeManager.getStartTime(Iteracao_T);
        //std::cout << "Active nodes: " << best.numActiveNodes << ", FitnessTrain: " << best.fitness << ", FitnessValidation: " << best.fitnessValidation  << std::endl;


        //printCircuit(&best, params);
        for (int i = 0; i < NUM_INDIV; i++){
            mutated_best = best;
            //mutateTopologyProbabilistic(&mutated_best, params, &seeds[i], 0);
            //mutateTopologyProbabilistic2(&mutated_best, params, seeds, 0, i)
            mutateSAM(&mutated_best, params, seeds);

            //evaluateCircuit(&mutated_best, training);
            //evaluateCircuitValidation(&mutated_best, validation);
            timeManager.getStartTime(Avaliacao_T);
            evaluateCircuit(&mutated_best, training);
            //evaluateCircuitValidationLinear(&mutated_best, validation);
            timeManager.getEndTime(Avaliacao_T);

            (*timeKernel) += timeManager.getElapsedTime(Avaliacao_T);;

            if(iterations%1000 == 0)
                std::cout << mutated_best.fitness << " ";
            if(mutated_best.fitness >= best_train.fitness){
                best_train = mutated_best;
            }

            /*if(mutated_best.fitnessValidation >= best_valid.fitnessValidation){
                best_valid = mutated_best;
            }
*/
            //std::cout << mutated_best.fitness << " ";
        }
        /*if(iterations%1000 == 0)
            std::cout << std::endl;*/
        best = best_train;
        //std::cout << "Best fitness  = " << best.fitness << std::endl;

        if(best.fitness == training->M) {
            //std::cout << "CGP achou o individuo factivel" << std::endl;
            //std::cout << "Geracao: " << iterations << std::endl;
            printChromosome(&best, params);
            printFile(&best, params, factivel_file);
            factivel = 1;
            break;
        }

        timeManager.getEndTime(Iteracao_T);

        timeManager.getElapsedTime(Iteracao_T);
        /*if(iterations%1000 == 0){
            printf("Generation %d:\n", iterations);
            printf("Time: %f\n", timeManager.getTotalTime(Iteracao_T));

        }*/

        iterations++;
    }
    (*timeIter) = timeManager.getTotalTime(Iteracao_T);


    return best;
}

Chromosome PCGP(Dataset* training, Parameters* params, OCLConfig* ocl, int *seeds, double* timeIter, double* timeKernel, std::ofstream& factivel_file){
    GPTime timeManager(4);
    //Chromosome *current_pop;
    //current_pop

    Chromosome best;
    Chromosome best_train;
    //Chromosome best_valid;
    Chromosome* population = new Chromosome[NUM_INDIV];
    ActiveChromosome* activePopulation = new ActiveChromosome[NUM_INDIV];
    CompactChromosome *compactPopulation = new CompactChromosome[NUM_INDIV];

    int factivel = 0;


    initializePopulation(population, params, &seeds[0]);

    int bestTrain = evaluatePopulation(population, training, 0);
    //int bestValid = evaluatePopulation(population, validation, 1);
    double kernelTime = 0;
    best_train = population[bestTrain];
    //best_valid = population[bestValid];
    best = best_train;

    if(best.fitness == training->M) {
        //std::cout << "CGP achou o indivíduo factível" << std::endl;
        //printChromosome(&best, params);
        printFile(&best, params, factivel_file);
        factivel = 1;
    }

    //std::cout << "Melhor da populacao: " << best.fitness << std::endl;
    //std::cout << "morreu aqui? 1 " << std::endl;
    ocl->writeReadOnlyBufers(params, seeds);

    int iterations = 0;
    while(stopCriteria(iterations) && (factivel != 1)) {
        timeManager.getStartTime(Iteracao_T);

        //std::cout << "Active nodes: " << best.numActiveNodes << ", FitnessTrain: " << best.fitness << ", FitnessValidation: " << best.fitnessValidation  << std::endl;

        for(int k = 0; k < NUM_INDIV; k++){
            population[k] = best;
            mutateSAM(&population[k], params, seeds);
            //mutateTopologyPoint(&population[k], params, seeds);
            //mutateTopologyProbabilistic2(&population[k], params, seeds, 1, k);
        }


#if DEFAULT
        ocl->writePopulationBuffer(population);
        ocl->finishCommandQueue();

        ocl->enqueueTrainKernel();
        //ocl->enqueueValidationKernel();
#elif COMPACT
        ocl->compactChromosome(population, compactPopulation);
        ocl->writePopulationCompactBuffer(compactPopulation);
        ocl->finishCommandQueue();

        ocl->enqueueTrainCompactKernel();
        ocl->enqueueValidationCompactKernel();
#elif IMAGE_R
        ocl->writeImageBuffer(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageKernel();
        ocl->enqueueEvaluationImageValidationKernel();
#elif IMAGE_RG
        ocl->writeImageBufferHalf(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageHalfKernel();
        ocl->enqueueEvaluationImageValidationHalfKernel();
#elif IMAGE_RGBA
        ocl->writeImageBufferQuarter(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageQuarterKernel();
        ocl->enqueueEvaluationImageValidationQuarterKernel();
#elif COMPACT_R
        ocl->writeImageBufferCompact(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageCompactKernel();
        ocl->enqueueEvaluationImageValidationCompactKernel();

#elif COMPACT_RG
        ocl->writeImageBufferHalfCompact(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageHalfCompactKernel();
        ocl->enqueueEvaluationImageValidationHalfCompactKernel();
#elif  COMPACT_RGBA
        ocl->writeImageBufferQuarterCompact(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageQuarterCompactKernel();
        ocl->enqueueEvaluationImageValidationQuarterCompactKernel();
#endif

        ocl->finishCommandQueue();
        kernelTime+= ocl->getKernelElapsedTimeTrain();
        //kernelTime+= ocl->getKernelElapsedTimeValid();


        //ocl->writeBestBuffer(&best);
        //ocl->writePopulationBuffer(population);

        //ocl->finishCommandQueue();

        //ocl->enqueueEvolveKernel();
        //ocl->finishCommandQueue();

        //ocl->readPopulationBuffer(population);
        //ocl->finishCommandQueue();
        //for(int k = 0; k < NUM_INDIV; k++){
        //    activateNodes(&population[k], params);
        //}

        //copyActiveNodes(population, activePopulation);
        //ocl->writePopulationActiveBuffer(activePopulation);
        //ocl->readPopulationBuffer(population);

        ocl->readFitnessBuffer();
        //ocl->readFitnessValidationBuffer();

        ocl->finishCommandQueue();

        /*for(int k = 0; k < NUM_INDIV; k++){
            std::cout << "Fitness do " << k << ": " << population[k].fitness << std::endl;
        }*/


        for(int k = 0; k < NUM_INDIV; k++){

            population[k].fitness = ocl->fitness[k];

            //std::cout << "ocl fitness k " << ocl->fitness[k] << std::endl;
            //population[k].fitnessValidation = ocl->fitnessValidation[k];


            /*std::cout << "Fitness de um individuo: ";
            std::cout << population[k].fitness << std::endl;*/

            if(population[k].fitness >= best_train.fitness){
                best_train = population[k];
               /* std::cout << "FITNESS best_train" << best_train.fitness << std::endl;
                std::cout << "FITNESS population[k]" << population[k].fitness << std::endl;*/
            }

            /*if(population[k].fitnessValidation >= best_valid.fitnessValidation){
                best_valid = population[k];
            }*/
        }

        /*if(iterations%1000 == 0)
            std::cout << std::endl;*/

        best = best_train;

       /* std::cout << "FITNESS best_train" << best_train.fitness << std::endl;
        std::cout << "FITNESS best" << best.fitness << std::endl;*/

        if(best.fitness == training->M) {
            //std::cout << "CGP achou o individuo factivel depois do kernel" << std::endl;
            //std::cout << "Geracao: " << iterations << std::endl;
            //printChromosome(&best, params);
            printFile(&best, params, factivel_file);
            factivel = 1;
            break;
        }

        timeManager.getEndTime(Iteracao_T);
        timeManager.getElapsedTime(Iteracao_T);

        /*if(iterations%1000 == 0){
            printf("Generation %d:\n", iterations);
            printf("Best fitness: %f\n", best.fitness);
            printf("Time: %f\n", timeManager.getTotalTime(Iteracao_T));
            printf("Kernel Time: %f\n", kernelTime);
        }*/
        iterations++;

    }


    (*timeIter) = timeManager.getTotalTime(Iteracao_T);
    (*timeKernel) = kernelTime;

    //ocl->readSeedsBuffer(seeds);
    ocl->finishCommandQueue();
    /* for(int i = 0; i < NUM_INDIV * ocl->maxLocalSize; i++){
         std::cout << seeds[i] << " ";
     }*/
    std::cout << std::endl;

    //printChromosome(&best, params);
    return best;
}

void printChromosome(Chromosome *c, Parameters *p) {
    for(int i = 0; i < MAX_NODES; i++) {
       if (c->nodes[i].active) {
           std::cout <<"---------------- comeco -----------------" << std::endl;
           std::cout <<"Indice do no " << i + p->N << std::endl;
           std::cout <<"No: " << std::endl;
           std::cout <<"Function: " << c->nodes[i].function << std::endl;
           std::cout <<"Inputs: " << c->nodes[i].inputs[0] << " "
           << c->nodes[i].inputs[1] << std::endl;
           std::cout <<"----------------   fim  -----------------" << std::endl;
       }
    }
    std::cout <<"Outputs: " << c->output[0] + p->N << std::endl;
}

void printFile(Chromosome *c, Parameters *p, std::ofstream& factivel_file) {
    for(int i = 0; i < MAX_NODES; i++) {
        if (c->nodes[i].active) {
            factivel_file << "Node" << i + p->N << " " << c->nodes[i].inputs[0]
            << " " << c->nodes[i].inputs[1] << " " <<  c->nodes[i].function << "\n";
        }
    }
    factivel_file << "Output " << c->output[0] + p->N << "\n";
}

Chromosome CGPDE_IN();

Chromosome CGPDE_OUT();

Chromosome PCGPDE_IN();

Chromosome PCGPDE_OUT();
