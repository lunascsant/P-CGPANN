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

    Dataset fullData;

    float** dataset;
    float** outputs;

    readDataset(params, &fullData, datasetFile);
    //readDataset(params, &dataset, &outputs, datasetFile);
    //printDataset(params, dataset, outputs);
    //test(params, dataset, outputs);

    Dataset* folds = generateFolds(&fullData);

    initializePopulation(current_pop, params, &seeds[0]);

    //best = current_pop[evaluatePopulation(current_pop, params, dataset, outputs)];
    best = current_pop[evaluatePopulation(current_pop, params, fullData.data, fullData.output)];
    new_best = best;
    //printPopulation(current_pop, params);

#if PARALLEL

    best = PCGP(best, &fullData, params, seeds);

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