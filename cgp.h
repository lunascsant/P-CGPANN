//
// Created by bruno on 04/02/2020.
//

#ifndef PCGP_CIRCUIT_H
#define PCGP_CIRCUIT_H

#include "utils.h"
#include "stack.h"
#include "OCLConfig.h"

void newNode(Chromosome* c, Parameters* params, unsigned int index, int* seed);
void activateNodes(Chromosome* c, Parameters* p);
void circuitGenerator(Chromosome* c, Parameters* params, int* seed);


void evaluateCircuit(Chromosome* c, Dataset* data);
void evaluateCircuitValidation(Chromosome* c, Dataset* data);

void evaluateCircuitLinear(Chromosome* c, Dataset* data);
void evaluateCircuitValidationLinear(Chromosome* c, Dataset* data);

void runCircuit(Chromosome* c, Dataset* dataset, int index, int validation);
void runCircuitLinear(Chromosome* c, Dataset* dataset, int index, int validation);

void initializePopulation(Chromosome* pop, Parameters* p, int* seed);
int evaluatePopulation(Chromosome* pop, Dataset* dataset, int validation);

Chromosome *mutate(Chromosome *c, Parameters *p, int *seed);
Chromosome *mutateTopologyProbabilistic(Chromosome *c, Parameters *p, int *seed, int type);
Chromosome *mutateTopologyProbabilisticActive(Chromosome *c, Parameters *p, int *seed, int type);
Chromosome *mutateSAM(Chromosome *c, Parameters *p, int *seed);
Chromosome *mutateTopologyPoint(Chromosome *c, Parameters *p, int *seed);


void CGP(Dataset *training,
         Parameters *params,
         int *seeds,
         double *timeIter,
         double *timeKernel,
         std::string caminhoArquivo);

//Chromosome PCGP(Dataset* training, Dataset* validation, Parameters* params, OCLConfig* ocl, int *seeds, double* timeIter, double* timeKernel);
void PCGP(Dataset* training,
          Parameters* params,
          OCLConfig* ocl, int *seeds,
          double* timeIter,
          double* timeKernel,
          std::string caminhoArquivo,
          std::string caminhoArquivoTime
          );

void printChromosome(Chromosome *c, Parameters *p);
void printFile(Chromosome *c, Parameters *p, std::ofstream& factivel_file);

Chromosome CGPDE_IN();
Chromosome CGPDE_OUT();

Chromosome PCGPDE_IN();
Chromosome PCGPDE_OUT();


#endif //PCGP_CIRCUIT_H
