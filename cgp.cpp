//
// Created by bruno on 04/02/2020.
//

#include "cgp.h"
#include "GPTime.h"
#include <vector>

void newNode(Chromosome* c, Parameters* params, unsigned int index, int* seed){
    /** set node function */
    c->nodes[index].function = params->functionSet[randomFunction(params, seed)];


    c->nodes[index].maxInputs = getFunctionInputs(c->nodes[index].function);
    //std::cout << "GetFunctionInputs: " << getFunctionInputs(c->nodes[index].function) << std::endl;
    //std::cout << "Function: " << c->nodes[index].function << " - Max Inputs: " << c->nodes[index].maxInputs << std::endl;

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
        //std::cout << " node index " << nodeIndex << std::endl;
        push(&s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            //std::cout << "Morre aqui no node " << node << std::endl;
            if(c->nodes[node].active == 0) {
                for (j = 0; j < MAX_ARITY; j++) {
                    //std::cout << "Morre aqui no j " << j << std::endl;
                    if (c->nodes[node].inputs[j] >= p->N) {
                        push(&s, c->nodes[node].inputs[j] - p->N);
                    }
                }
                c->activeNodes[c->numActiveNodes] = node;
                c->nodes[node].active = 1;
                c->numActiveNodes++;
               // std::cout << "Morre aqui no if" << std::endl;
            }

        }
    }
    sortActiveArray(c->activeNodes, c->numActiveNodes);
}

int get_num_transistors(int gate)
{
    switch (gate)
    {
        case 10: //AND
            return 2;
            break;
        case 11: //OR
            return 2;
            break;
        case 14: //NOT
            return 1;
            break;
        case 15: //NAND
            return 2;
            break;
        case 16: //NOR
            return 1;
            break;
        case 12: //XOR
            return 3;
            break;
        case 17: //XNOR
            return 4;
            break;
        default:
            std::cout << "Gate code unknow!\n";
            exit(1);
            break;
    }
}

void count_num_transistors_individual(Chromosome *c)
{
    int temp = 0;
    for(int j = 0; j < MAX_NODES; j++)
    {
        if (c->nodes[j].active == 1)
        {
            temp += get_num_transistors(c->nodes[j].function);
        }
    }
    c->numTransistors = temp;
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
    c->numTransistors = 0;
    c->numActiveNodes = 0;
    count_num_transistors_individual(c);
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

    int r1, r2;



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
           // std::cout << "AND " << result << std::endl;
            break;
        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
           // std::cout << "OR " << result << std::endl;
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
            //r1 = popEx(exStack);
            //r2 = popEx(exStack);
            //std::cout << "valor r1: " << r1 << " valor r2: " << r2 << std::endl;


            //std::cout << "XOR " << result << std::endl;
            break;
        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
            //std::cout << "NAND " << result << std::endl;
            break;
        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
            //std::cout << "NOR " << result << std::endl;
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
            //std::cout << "XNOR " << result << std::endl;
            break;
        case NOT:
            result = 0;
            if(popEx(exStack) == 0){
                result = 1;
            }
           // std::cout << "NOT " << result << std::endl;
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
            //std::cout << "Function: " << c->nodes[node].function << " - Max inputs: " << c->nodes[node].maxInputs << std::endl;
            /*if(c->nodes[node].function == 14){
                c->nodes[node].maxInputs = 1;
            }
            else{
                c->nodes[node].maxInputs = 2;
            }*/

            for (int j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                //std::cout << "J ATUAL: " << j << std::endl;
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
                    //std::cout << "Input 0: " << c->nodes[node].inputs[0] << " - Input 1: " << c->nodes[node].inputs[1] << " - Function: " << c->nodes[node].function << std::endl;
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
            //std::cout << "Index: " << index << " i: " << i << " Valor desejado: " << dataset->output[index][i] << " - Valor obtido " << executionOut[i] << std::endl;
            (c->fitness)++;
        }
        //std::cout << "- FIM AVALIACAO -" << std::endl;
        //std::cout << "Dataset " << dataset->output[index][i] << std::endl;
        //std::cout << "Execution out: " << executionOut[i] << std::endl;
    }

    /*std::cout << "Fitness " << c->fitness << std::endl;*/

    /*if(predictedClass == correctClass) {
        if(validation == 1){
            (c->fitnessValidation)++;
        } else {
            (c->fitness)++;
        }

    }*/
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

void evaluatePopulation(Chromosome* pop, Dataset* dataset, int validation, int* bestInd){
    int i, j;
    float bestFitness = 0;
    unsigned int bestNumTransistors = 99999;
    int bestIndex = -1;

    for(j = 0; j < NUM_INDIV; j++) {
        evaluateCircuit(&pop[j], dataset);
        if(pop[j].fitness > bestFitness){
            bestFitness = pop[j].fitness;
            bestNumTransistors = pop[j].numTransistors;
            bestIndex = j;
        }
    }

    *bestInd = bestIndex;

    int feasibles = 0;

    for(i = 0; i < NUM_INDIV_POP; i++) {
        if (pop[i].fitness == (dataset->M * dataset->O)) {
            feasibles += 1;
        }
    }

    // VERIFICAR QUESTAO DE MAIS DE UM TER MESMO TANTO DE TRANSISTORS
    std::vector<int> equalFitness;
    std::vector<int> equalTransistors;
    int indBest = -1;

    if(feasibles > 0) {
        if(feasibles == 1) {
            for(i = 0; i < NUM_INDIV_POP; i++) {
                if(pop[i].fitness == (dataset->M * dataset->O)){
                    bestIndex = i;
                }
            }
        } else if (feasibles > 1) {
            for(i = 0; i < NUM_INDIV_POP; i++) {
                if(pop[i].fitness == (dataset->M * dataset->O)){
                    if(pop[i].numTransistors < bestNumTransistors) {
                        bestNumTransistors = pop[i].numTransistors;
                        bestIndex = i;
                        equalTransistors.clear();
                        equalTransistors.push_back(i);
                    } else if (pop[i].numTransistors == bestNumTransistors) {
                        equalTransistors.push_back(i);
                    }
                }
            }

            if(!equalTransistors.empty()) {
                if(equalTransistors.size() == 1) {
                    bestIndex = equalTransistors.at(0);
                } else {
                    indBest = rand() % (equalTransistors.size() - 1);
                    bestIndex = equalTransistors.at(indBest);
                }
            }
        }
    } else {
        for(i = 0; i < NUM_INDIV_POP; i++) {
            if(pop[i].fitness > bestFitness) {
                bestFitness = pop[i].fitness;
                bestNumTransistors = pop[i].numTransistors;
                bestIndex = i;
                equalFitness.clear();
                equalFitness.push_back(i);
            } else if (pop[i].fitness == bestFitness) {
                equalFitness.push_back(i);
            }
        }

        equalTransistors.clear();

        if(!equalFitness.empty()) {
            if(equalFitness.size() == 1) {
                bestIndex = equalFitness.at(0);
            } else {
                indBest = rand() % (equalFitness.size() - 1);
                bestIndex = equalFitness.at(indBest);
            }
        }
    }

    *bestInd = bestIndex;
    equalFitness.clear();
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

    count_num_transistors_individual(c);
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

    count_num_transistors_individual(c);
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

    count_num_transistors_individual(c);
    activateNodes(c, p);
    return  c;
}

Chromosome *mutateSAM(Chromosome *c, Parameters *p, int *seed) {
    //std::cout << "inicio mutacao" << std::endl;
    int i, j, inputOrFunction, nodeOrOutput, indOutputToBeModified;
    int activeSelected = 0;

    nodeOrOutput = randomInterval(0, MAX_NODES + p->O - 1, seed);

    if (nodeOrOutput > MAX_NODES - 1) {
        activeSelected = 1;
        indOutputToBeModified = randomInterval(0, MAX_OUTPUTS, seed);
        c->output[indOutputToBeModified] = randomInterval(0, MAX_NODES - 1, seed);
    } else {
        while(activeSelected == 0) {
            i = randomInterval(0, MAX_NODES - 1, seed);
            //std::cout << "ind no " << i << " active node " << c->nodes[i].active << std::endl;
            if(c->nodes[i].active == 1) {
                activeSelected = 1;
            }
            inputOrFunction = randomInterval(0, 1, seed);
            if (!inputOrFunction) {
                c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
                c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
            } else {
                j = randomInterval(0, 1, seed);
                c->nodes[i].inputs[j] = randomInput(p, i, seed);
            }
        }
    }



    count_num_transistors_individual(c);
    activateNodes(c, p);
    //std::cout << "fim mutacao" << std::endl;
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

    count_num_transistors_individual(c);
    activateNodes(c, p);
    return  c;
}

void personalStack(Chromosome *c, SomoAuxStruct *somo_aux, int node, int output, int n_i) {
    int function = c->nodes[node].function;
    int inp0 = c->nodes[node].inputs[0];
    int inp1 = c->nodes[node].inputs[1];

    if(node >= n_i) {
        if(function == NOT) {
            if(std::find(somo_aux->myStack[output].begin(), somo_aux->myStack[output].end(), inp0) != somo_aux->myStack[output].end()) {
                somo_aux->myStack[output].push_back(inp0);
            }
            personalStack(c, somo_aux, inp0, output, n_i);
        } else {
            if(std::find(somo_aux->myStack[output].begin(), somo_aux->myStack[output].end(), inp0) != somo_aux->myStack[output].end()) {
                somo_aux->myStack[output].push_back(inp0);
            }
            if(std::find(somo_aux->myStack[output].begin(), somo_aux->myStack[output].end(), inp1) != somo_aux->myStack[output].end()) {
                somo_aux->myStack[output].push_back(inp1);
            }
            personalStack(c, somo_aux, inp0, output, n_i);
            personalStack(c, somo_aux, inp1, output, n_i);
        }
    }
        
}

void simulateI0I1(Chromosome *c, SomoAuxStruct *somo_aux, int out, int node_cind, int input_eind, int n_i, Dataset *data) {
    int inpv, inpf, function;
    
    for(j = 0; j < 2; j++) {
        for(int i = 0; i < somo_aux->myStack[out].size(); i++) {
            if(somo_aux->myStack[out][i] >= node_cind) {
                if(somo_aux->myStack[out][i] == node_cind) {
                    inpv = c->nodes[somo_aux->myStack[out][i] - n_i].inputs[input_eind];
                    if(input_eind == 0) {
                        inpf = c->nodes[somo_aux->myStack[out][i] - n_i].inputs[1];
                    } else {
                        inpf = c->nodes[somo_aux->myStack[out][i] - n_i].inputs[0];
                    }

                    function = c->nodes[somo_aux->myStack[out][i] - n_i].function;
                    
                    switch(function) {
                        case AND: 
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = somo_aux->truthTable[inpf][k] && j;
                            }
                            break;
                        case OR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = somo_aux->truthTable[inpf][k] || j;
                            }
                            break;
                        case NOR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !(somo_aux->truthTable[inpf][k] || j);
                            }
                            break;
                        case NOT:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !somo_aux->truthTable[inpf][k];
                            }
                            break;
                        case NAND:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !(somo_aux->truthTable[inpf][k] && j);
                            }
                            break;
                        case XOR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = (somo_aux->truthTable[inpf][k] && !j) || (!somo_aux->truthTable[inpf][k] && j);
                            }
                            break;
                        case XNOR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !((somo_aux->truthTable[inpf][k] && !j) || (!somo_aux->truthTable[inpf][k] && j));
                            }
                            break;
                        default:
                            break;
                    }
                    
                } else {
                    inpv = c->nodes[somo_aux->myStack[out][i] - n_i].inputs[0];
                    inpf = c->nodes[somo_aux->myStack[out][i] - n_i].inputs[1];
                    
                    function = c->nodes[somo_aux->myStack[out][i] - n_i].function;
                    
                    switch(function) {
                        case AND: 
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = somo_aux->truthTable[inpf][k] && somo_aux->truthTable[inpv][k];
                            }
                            break;
                        case OR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = somo_aux->truthTable[inpf][k] || somo_aux->truthTable[inpv][k];
                            }
                            break;
                        case NOR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !(somo_aux->truthTable[inpf][k] || somo_aux->truthTable[inpv][k]);
                            }
                            break;
                        case NOT:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !somo_aux->truthTable[inpf][k];
                            }
                            break;
                        case NAND:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !(somo_aux->truthTable[inpf][k] && somo_aux->truthTable[inpv][k]);
                            }
                            break;
                        case XOR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = (somo_aux->truthTable[inpf][k] && !somo_aux->truthTable[inpv][k]) || (!somo_aux->truthTable[inpf][k] && somo_aux->truthTable[inpv][k]);
                            }
                            break;
                        case XNOR:
                            for(int k = 0; k < data->M; k++) {
                                somo_aux->truthTable[node_cind][k] = !((somo_aux->truthTable[inpf][k] && !somo_aux->truthTable[inpv][k]) || (!somo_aux->truthTable[inpf][k] && somo_aux->truthTable[inpv][k]));
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        // tabela verdade preenchida
        for(int t = 0; t < data->M; t++) {
            somo_aux->eSimulation[out][j].push_back(somo_aux->truthTable[somo_aux->myStack[out][myStack[out].size() - 1]][t]);
        } 
    }
}

void calculateReq(SomoAuxStruct *somo_aux, int out) {
    for(int i = 0; i < somo_aux->outputs_transpose[out].size(); i++) {
        if(somo_aux->eSimulation[out][0][i] == somo_aux->outputs_transpose[out][i]) {
            if(somo_aux->eSimulation[out][0][i] == somo_aux->eSimulation[out][1][i]) {
                somo_aux->req[out].push_back(9);
            } else {
                somo_aux->req[out].push_back(0);
            }
        } else {
            somo_aux->req[out].push_back(1);
        }
    }
}

int *identifyBestNode(int node_cind, int input_eind, Chromosome *c, Dataset* data, int *seed) {
    std::vector<int> connection_candidates;

    for(int i = 0; i < node_cind; i++) {
        connection_candidates.push_back(i);
    }

    SomoAuxStruct somo_aux;
    somo_aux.truthTable = new unsigned short int[data->N + MAX_NODES];
    somo_aux.score = new int[MAX_NODES];

    for(int i = 0; i < data->N + MAX_NODES; i++) {
        somo_aux.truthTable[i] = new unsigned short int[data->M];
    }

    for(int i = 0; i < data->N; i++) {
        for(int j = 0; j < data->M; j++) {
            somo_aux.truthTable[i][j] = data->data[j][i];
        }
    }

    // calculando a tabela verdade dos nós que vem antes do nó escolhido
    for(int i = 0; i < connection_candidates.size(); i++) {
        int function = c->nodes[connection_candidates[i]].function;
        int inp0 = c->nodes[connection_candidates[i]].inputs[0];
        int inp1 = c->nodes[connection_candidates[i]].inputs[1];
        
        switch(function) {
            case AND: 
                for(int k = 0; k < data->M; k++) {
                    somo_aux.truthTable[connection_candidates[i] + data-> N][k] = somo_aux.truthTable[inp0][k] && somo_aux.truthTable[inp1][k];
                }
                break;
            case OR:
                for(int k = 0; k < data->M; k++) {
                    somo_aux.truthTable[connection_candidates[i] + data-> N][k] = somo_aux.truthTable[inp0][k] || somo_aux.truthTable[inp1][k];
                }
                break;
            case NOR:
                for(int k = 0; k < data->M; k++) {
                    somo_aux.truthTable[connection_candidates[i] + data-> N][k] = !(somo_aux.truthTable[inp0][k] || somo_aux.truthTable[inp1][k]);
                }
                break;
            case NOT:
                for(int k = 0; k < data->M; k++) {
                    somo_aux.truthTable[connection_candidates[i] + data-> N][k] = !somo_aux.truthTable[inp0][k];
                }
                break;
            case NAND:
                for(int k = 0; k < data->M; k++) {
                    somo_aux.truthTable[connection_candidates[i] + data-> N][k] = !(somo_aux.truthTable[inp0][k] && somo_aux.truthTable[inp1][k]);
                }
                break;
            case XOR:
                for(int k = 0; k < data->M; k++) {
                    somo_aux.truthTable[connection_candidates[i] + data-> N][k] = (somo_aux.truthTable[inp0][k] && !somo_aux.truthTable[inp1][k]) || (!somo_aux.truthTable[inp0][k] && somo_aux.truthTable[inp1][k]);
                }
                break;
            case XNOR:
                for(int k = 0; k < data->M; k++) {
                    somo_aux.truthTable[connection_candidates[i] + data-> N][k] = !((somo_aux.truthTable[inp0][k] && !somo_aux.truthTable[inp1][k]) || (!somo_aux.truthTable[inp0][k] && somo_aux.truthTable[inp1][k]));
                }
                break;
            default:
                break;
        }
        
    }

    for(int i = 0; i < data->O; i++) {
        for(int j = 0; j < data->M; j++) {
            somo_aux.outputs_transpose[i].push_back(data->output[j][i]);
        }
    }

    for(int i = 0; i < MAX_OUTPUTS; i++) {
        personalStack(c, &somo_aux, c->output[i], i, data->N);
        if(std::find(somo_aux.myStack[i].begin(), somo_aux.myStack[i].end(), node_cind) != somo_aux.myStack[i].end()) {
            // faz simulacao 0 e 1 para saida i
            std::sort(somo_aux.myStack[i].begin(), somo_aux.myStack[i].end()); 
            simulateI0I1(c, &somo_aux, i, node_cind + data->N, input_eind, data->N, data);
            calculateReq(&somo_aux, i);
        }
    }

    int score;
    for(int j = 0; j < node_cind + data->N; j++) {
        score = 0;
        for(int i = 0; i < MAX_OUTPUTS; i++) {
            for(int k = 0; k < somo_aux.req[i].size(); k++) {
                if(somo_aux.truthTable[j][k] == somo_aux.req[i][k]) {
                    score++;
                }
            }
        }

        somo_aux.score.push_back(score);
    }

    std::vector<int> best_nodes;

    int max = (int) *std::max_element(somo_aux.score.begin(), somo_aux.score.end());

    for(int i = 0; i < somo_aux.score.size(); i++) {
        if(somo_aux.score[i] == max) {
            best_nodes.push_back(i);
        }
    }

    int ind_best = randomInterval(0, best_nodes.size(), seed);
    return ind_best;
}

Chromosome *mutateSOMO(Chromosome *c, Parameters *p, int *seed, Dataset* data) {
    int i, j;
    float randomProb;
    int node_cind, input_eind, bestNode;
    std::vector<int> active_nodes;


    for(i = 0; i < MAX_NODES - 1; i++) {
        if(c->nodes[i].active) {
            active_nodes.push_back(i)
        }
    }

    node_cind = randomInterval(0, active_nodes.size() - 1, seed);
    
    randomProb = randomInterval(0, 1);
    if(randomProb < PF) {
        c->nodes[node_cind].function = p->functionSet[randomFunction(p, seed)];
        c->nodes[node_cind].maxInputs = getFunctionInputs(c->nodes[i].function);
    } else {
        std::vector<int> mutatedInactiveNodes;

        for(int i = 0; i < PQ * MAX_NODES; i++) {
            j = randomInterval(0, MAX_NODES - 1, seed);
            while(c->nodes[j].active && std::find(mutatedInactiveNodes.begin(), mutatedInactiveNodes.end(), j) != mutatedInactiveNodes.end()) {
                j = randomInterval(0, MAX_NODES - 1, seed);
            }

            mutatedInactiveNodes.push_back(j);
            c->nodes[j].function = p->functionSet[randomFunction(p, seed)];
            c->nodes[j].maxInputs = getFunctionInputs(c->nodes[j].function);

            for(int k = 0; k < MAX_ARITY - 1; k++) {
                c->nodes[j].inputs[k] = randomInput(p, j, seed);
            }
        }
        
        input_eind = randomInterval(0, MAX_ARITY, seed);

        // já faz a conexão do nó c com o n ali dentro
        bestNode = identifyBestNode(node_cind, input_eind, c, data);
        c->nodes[node_cind].inputs[input_eind] = bestNode;
    }

}



Chromosome*
CGP(Dataset *training, Parameters *params, int *seeds, double *timeIter, double *timeKernel, std::ofstream& factivel_file) {
    GPTime timeManager(4);
    Chromosome *current_pop;
    current_pop = new Chromosome[NUM_INDIV_POP];

    Chromosome best;
    Chromosome best_train;
    Chromosome mutated_best;

    initializePopulation(current_pop, params, &seeds[0]);

    int bestTrain;
    evaluatePopulation(current_pop, training, 0, &bestTrain);

    unsigned int bestNumTransistors = 99999;

    int feasibles = 0;

    best_train = current_pop[bestTrain];

    best = best_train;

    std::vector<int> equalFitness;
    std::vector<int> equalTransistors;
    std::vector<Chromosome> feasiblesArray;

    int indBest = -1;
    int iterations = 0;

    while(stopCriteria(iterations)) {
        timeManager.getStartTime(Iteracao_T);

        for(int k = 0; k < NUM_INDIV_POP; k++){
            current_pop[k] = best;
            mutateSAM(&current_pop[k], params, seeds);
        }


        for(int i = 0; i < NUM_INDIV_POP; i++) {
            timeManager.getStartTime(Avaliacao_T);
            evaluateCircuit(&current_pop[i], training);
            timeManager.getEndTime(Avaliacao_T);

            (*timeKernel) += timeManager.getElapsedTime(Avaliacao_T);
        }

        for(int i = 0; i < NUM_INDIV_POP; i++) {
            if (current_pop[i].fitness == params->M) {
                feasibles += 1;
                feasiblesArray.push_back(current_pop[i]);
            }
        }

        // ACHEI FACTIVEL SAIO DO LAÇO
        if(feasibles > 0) {
            if (feasibles == 1) {
                best = best_train;
                break;
            } else {
                for (int i = 0; i < feasiblesArray.size(); i++) {
                    if (feasiblesArray[i].numTransistors < bestNumTransistors) {
                        bestNumTransistors = feasiblesArray[i].numTransistors;
                        best = feasiblesArray[i];
                        equalTransistors.clear();
                        equalTransistors.push_back(i);
                    } else if (feasiblesArray[i].numTransistors == bestNumTransistors) {
                        equalTransistors.push_back(i);
                    }
                }

                if (!equalTransistors.empty()) {
                    if (equalTransistors.size() == 1) {
                        best = feasiblesArray[equalTransistors.at(0)];
                    } else {
                        indBest = rand() % (equalTransistors.size() - 1);
                        best = feasiblesArray[equalTransistors.at(indBest)];
                    }
                }

                break;
            }
        }

        for(int i = 0; i < NUM_INDIV_POP; i++) {
            if(current_pop[i].fitness > best_train.fitness){
                best_train = current_pop[i];
                bestNumTransistors = current_pop[i].numTransistors;
                equalFitness.clear();
                equalFitness.push_back(i);
            } else if (current_pop[i].fitness == best_train.fitness) {
                equalFitness.push_back(i);
            }
        }


        if(!equalFitness.empty()) {
            if(equalFitness.size() == 1) {
                bestNumTransistors = current_pop[equalFitness.at(0)].numTransistors;
                best_train = current_pop[equalFitness.at(0)];
            } else {
                for(int i = 0; i < equalFitness.size(); i++) {
                    if(current_pop[i].numTransistors < bestNumTransistors) {
                        bestNumTransistors = current_pop[i].numTransistors;
                        best_train = current_pop[i];
                        equalTransistors.clear();
                        equalTransistors.push_back(i);
                    } else if (current_pop[i].numTransistors == bestNumTransistors) {
                        equalTransistors.push_back(i);
                    }
                }

                if(!equalTransistors.empty()) {
                    if(equalTransistors.size() == 1) {
                        best_train = current_pop[equalTransistors.at(0)];
                    } else {
                        indBest = rand() % (equalTransistors.size() - 1);
                        best_train = current_pop[equalTransistors.at(indBest)];
                    }
                }
            }
        }


        bestNumTransistors = 99999;
        equalFitness.clear();
        equalTransistors.clear();
        indBest = -1;

        best = best_train;

        timeManager.getEndTime(Iteracao_T);

        timeManager.getElapsedTime(Iteracao_T);

        iterations++;
    }

    (*timeIter) = timeManager.getTotalTime(Iteracao_T);

    return best;
}

Chromosome PCGP(Dataset* training, Parameters* params, OCLConfig* ocl, int *seeds, double* timeIter, double* timeKernel, std::ofstream& factivel_file){
    GPTime timeManager(4);

    Chromosome best;
    Chromosome best_train;
    Chromosome* population = new Chromosome[NUM_INDIV_POP];
    ActiveChromosome* activePopulation = new ActiveChromosome[NUM_INDIV_POP];
    CompactChromosome *compactPopulation = new CompactChromosome[NUM_INDIV_POP];

    initializePopulation(population, params, &seeds[0]);

    int bestTrain;

    evaluatePopulation(population, training, 0, &bestTrain);


    int feasibles = 0;

    double kernelTime = 0;

    best_train = population[bestTrain];

    best = best_train;

    ocl->writeReadOnlyBufers(params);

    std::vector<int> equalFitness;
    std::vector<int> equalTransistors;
    std::vector<Chromosome> feasiblesArray;

    int indBest = -1;
    int iterations = 0;

    while(stopCriteria(iterations)) {
        timeManager.getStartTime(Iteracao_T);

        int group = -1;
        for(int k = 0; k < NUM_INDIV; k++) {
            if(k % NUM_EXECUTIONS == 0) {
                group++;
            }
            population[k] = best[group];
            mutateSOMO(&population[k], params, seeds, training);
            //mutateSAM(&population[k], params, seeds);
        }

#if DEFAULT
        ocl->writePopulationBuffer(population);
        ocl->finishCommandQueue();

        ocl->enqueueTrainKernel();
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
        ocl->readFitnessBuffer();

        ocl->finishCommandQueue();

        for(int i = 0; i < NUM_INDIV_POP; i++) {
            population[i].fitness = ocl->fitness[i];
        }

        /*if(best.fitness == (params->M * params->O)) {
            std::cout << "Best Fitness " << best.fitness << std::endl;
            std::cout << "Best Transistors " << best.numTransistors << std::endl;
            for(int i = 0; i < NUM_INDIV_POP; i++) {
                std::cout << "Indiv " << i << " Fitness " << population[i].fitness << std::endl;
                std::cout << "Indiv " << i << " Transistors " << population[i].numTransistors << std::endl;
            }

            std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << std::endl;
        }*/

        feasiblesArray.clear();
        feasibles = 0;

        for(int i = 0; i < NUM_INDIV_POP; i++) {
            if (population[i].fitness == (params->M * params->O)) {
                feasibles += 1;
                feasiblesArray.push_back(population[i]);
            }
        }


        if(feasibles > 0) {
            if (feasibles == 1) {
                if(best.numTransistors > feasiblesArray.at(0).numTransistors) {
                    best = feasiblesArray.at(0);
                }
            } else {
                for (int i = 0; i < feasiblesArray.size(); i++) {
                    if (feasiblesArray.at(i).numTransistors < best.numTransistors) {
                        best = feasiblesArray.at(i);
                        equalTransistors.clear();
                        equalTransistors.push_back(i);
                    } else if (feasiblesArray.at(i).numTransistors == best.numTransistors) {
                        equalTransistors.push_back(i);
                    }
                }

                if (!equalTransistors.empty()) {
                    if (equalTransistors.size() == 1) {
                        best = feasiblesArray.at(equalTransistors.at(0));
                    } else {
                        indBest = rand() % (equalTransistors.size() - 1);
                        best = feasiblesArray.at(equalTransistors.at(indBest));
                    }
                }

                // IMPRIMIR FACTIVEL SO O PRIMEIRO NO ARQUIVO
            }
        }

        // std::cout << "feasibles: " << feasibles << std::endl;

        if(feasibles == 0) {
            for(int i = 0; i < NUM_INDIV_POP; i++) {
                if(population[i].fitness > best.fitness){
                    best = population[i];
                    equalFitness.clear();
                    equalFitness.push_back(i);
                } else if (population[i].fitness == best.fitness) {
                    equalFitness.push_back(i);
                }
            }


            if(!equalFitness.empty()) {
                if(equalFitness.size() == 1) {
                    best = population[equalFitness.at(0)];
                } else {
                    indBest = rand() % (equalFitness.size() - 1);
                    best = population[equalFitness.at(indBest)];
                }
            }
        }

        equalFitness.clear();
        equalTransistors.clear();
        indBest = -1;

        timeManager.getEndTime(Iteracao_T);
        timeManager.getElapsedTime(Iteracao_T);

        if(iterations % 50000 == 0 || iterations == 1) {
            std::cout << "Iter " << iterations << std::endl;
            std::cout << "Fitness do melhor " << best.fitness << std::endl;
            std::cout << "Transistores do melhor " << best.numTransistors << std::endl;
        }
        iterations++;

    }

    (*timeIter) = timeManager.getTotalTime(Iteracao_T);
    (*timeKernel) = kernelTime;

    ocl->finishCommandQueue();

    std::cout << std::endl;

    return best;
}

void printChromosome(Chromosome *c, Parameters *p) {
    for(int i = 0; i < MAX_NODES; i++) {
        //std::cout << c->nodes[i].inputs[0] << " " << c->nodes[i].inputs[1] << " " << c->nodes[i].function << std::endl;
       if (c->nodes[i].active) {
           /*std::cout <<"---------------- comeco -----------------" << std::endl;
           std::cout <<"Indice do no " << i + p->N << std::endl;
           std::cout <<"No: " << std::endl;
           std::cout <<"Function: " << c->nodes[i].function << std::endl;
           std::cout <<"Inputs: " << c->nodes[i].inputs[0] << " "
           << c->nodes[i].inputs[1] << std::endl;
           std::cout <<"----------------   fim  -----------------" << std::endl;*/
           std::cout << "No" << i + p->N << " - " << c->nodes[i].inputs[0] << " " << c->nodes[i].inputs[1] << " " << c->nodes[i].function << std::endl;
       }
    }
    std::cout <<"Outputs: " << c->output[0] + p->N << std::endl;//
    //std::cout << "Output: " << c->output[0] << std::endl;
    std::cout << "- FIM -" << std::endl;
}

void printFile(Chromosome *c, Parameters *p, std::ofstream& factivel_file) {
    for(int i = 0; i < MAX_NODES; i++) {
        if (c->nodes[i].active) {
            factivel_file << "Node" << i + p->N << " " << c->nodes[i].inputs[0]
            << " " << c->nodes[i].inputs[1] << " " <<  c->nodes[i].function << "\n";
        }
    }
    factivel_file << "Output " << c->output[0] + p->N << "\n";
    factivel_file << "\n";
}

Chromosome CGPDE_IN();

Chromosome CGPDE_OUT();

Chromosome PCGPDE_IN();

Chromosome PCGPDE_OUT();
