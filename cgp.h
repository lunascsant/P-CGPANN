//
// Created by bruno on 04/02/2020.
//

#ifndef PCGP_CIRCUIT_H
#define PCGP_CIRCUIT_H

#include "utils.h"
#include "stack.h"
/**
 * How the Chromosome is build:
 *
 *           _____________________________________________________________
 * nodes:   |( F0 | I0 | I1 )|( F2 | I0' | I1' )| ... |( Fn | I0n | I1n )|
 *          -------------------------------------------------------------
 *          Fn -> Value coded with a function from the functions set
 *          I0, I1 -> Inputs of the function. Can be wither a Variable from the dataset or any node with a smaller index
 *          -> This way, each node occupies 3 spaces in the array.
 *
 *           _____________________
 * outputs: | O0 | O1 | ... | On |
 *          ---------------------
 *          On -> index of the node from which the output is taken
 *
 * fitness: sum of
 */

typedef struct
{
    //unsigned int inputsEvaluated;
    unsigned int function;
    unsigned int maxInputs;
    unsigned int inputs[MAX_ARITY];
    float inputsWeight[MAX_ARITY];
    //float output;
    int active;

} Node;

typedef struct
{
    Node nodes[MAX_NODES];
    unsigned int output[MAX_OUTPUTS];
    unsigned int numActiveNodes;
    float fitness;
    float fitnessValidation;
} Chromosome;



void newNode(Chromosome* c, Parameters* params, unsigned int index, int* seed);

void activateNodes(Chromosome* c, Parameters* p);

void circuitGenerator(Chromosome* c, Parameters* params, int* seed);

void evaluateCircuit(Chromosome* c, Parameters* p, float** data, float** out);
void evaluateCircuit(Chromosome* c, Dataset* data);
void evaluateCircuitValidation(Chromosome* c, Parameters* p, float** data, float** out);

void runCircuit(Chromosome* c, Parameters* p, float* data, float* out, int validation);
void runCircuit(Chromosome* c, Dataset* datasettt, int index, int validation);

void initializePopulation(Chromosome* pop, Parameters* p, int* seed);

int evaluatePopulation(Chromosome* pop, Parameters* p, float** data, float** out);

Chromosome *mutate(Chromosome *c, Parameters *p, int *seed);
Chromosome *mutateTopologyProbabilistic(Chromosome *c, Parameters *p, int *seed, int type);
Chromosome *mutateTopologyProbabilisticActive(Chromosome *c, Parameters *p, int *seed, int type);
Chromosome *mutateTopologyPoint(Chromosome *c, Parameters *p, int *seed);

void test(Parameters* p, float** data, float** out);

Chromosome CGP(Chromosome best, float** dataset, float** outputs, Parameters* params, int *seeds);
Chromosome CGP(Dataset* training, Dataset* validation, Parameters* params, int *seeds);

Chromosome PCGP(Dataset* data, Parameters* params, int *seeds);
Chromosome PCGP(Dataset* training, Dataset* validation, Parameters* params, int *seeds);

Chromosome CGPDE_IN();

Chromosome CGPDE_OUT();

Chromosome PCGPDE_IN();

Chromosome PCGPDE_OUT();


#endif //PCGP_CIRCUIT_H
