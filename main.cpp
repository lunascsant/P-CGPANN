#include <iostream>

#include "cgp.h"
#include "utils.h"
#include "GPTime.h"


int main(int argc, char** argv) {
    char* datasetFile = argv[1];

    GPTime timeManager(4);
    timeManager.getStartTime(Total_T);

    Parameters *params;
    params = new Parameters;

    Chromosome *current_pop;
    current_pop = new Chromosome[NUM_INDIV];

    int* seeds;
    seeds = new int [NUM_INDIV];
    srand(SEED);

    /*random seeds used in parallel code*/
    for(int i = 0; i < NUM_INDIV; i++){
        seeds[i] = rand();
    }

    Chromosome best;
    Chromosome new_best;
    Chromosome newBest[NUM_INDIV];

    Chromosome mutated_best;

    float** dataset;
    float** outputs;

    Dataset data;

    readDataset(params, &dataset, &outputs, datasetFile);
    //printDataset(params, dataset, outputs);
    //test(params, dataset, outputs);

    ///transposicao de dados
    float* transposeDataset;
    float* transposeOutputs;

    transposeDataset = new float [params->M * params->N];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    unsigned int pos = 0;
    std::cout << "Transpondo dados..." << std::endl;
    for(int j = 0; j < params->N; ++j ){
        for(int i = 0; i < params->M; ++i ){
            transposeDataset[pos++] = dataset[i][j];
        }
    }

    transposeOutputs = new float [params->M * params->O];
    //transposição necessária para otimizar a execução no opencl com acessos sequenciais à memória
    pos = 0;
    std::cout << "Transpondo outputs..." << std::endl;
    for(int j = 0; j < params->O; ++j ){
        for(int i = 0; i < params->M; ++i ){
            transposeOutputs[pos++] = outputs[i][j];
        }
    }

    initializePopulation(current_pop, params, &seeds[0]);

    best = current_pop[evaluatePopulation(current_pop, params, dataset, outputs)];
    new_best = best;
    //printPopulation(current_pop, params);

#if PARALLEL

    best = PCGP(best, transposeDataset, transposeOutputs, params, seeds);

#else

    best = CGP(best, dataset, outputs, params, seeds);

#endif

    //printPopulation(current_pop, params);
    timeManager.getEndTime(Total_T);

    //printCircuit(&best, params);
    std::cout << "Best fitness  = " << best.fitness << std::endl;

    std::cout << "Total time  = " << timeManager.getElapsedTime(Total_T) << std::endl;
    std::cout << "Iter time  = " << timeManager.getTotalTime(Iteracao_T) << std::endl;


    delete[] current_pop;
    delete params;

    return 0;
}