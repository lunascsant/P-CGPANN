#include <iostream>

#include "cgp.h"
#include "utils.h"
#include "GPTime.h"
#include "OCLConfig.h"


int main(int argc, char** argv) {
    char* datasetFile = argv[1];
    FILE *f_CGP = fopen("./results/cgpann.txt", "w");
    fprintf(f_CGP, "i,\tj,\taccuracy\n");

    GPTime timeManager(4);
    timeManager.getStartTime(Total_T);

    Parameters *params;
    params = new Parameters;

    int* seeds;
    seeds = new int [NUM_INDIV];
    srand(SEED);

    /*random seeds used in parallel code*/
    for(int i = 0; i < NUM_INDIV; i++){
        seeds[i] = rand();
    }

    Chromosome executionBest;

    Dataset fullData;

    readDataset(params, &fullData, datasetFile);

    int trainSize, validSize, testSize;
    calculateDatasetsSize(&fullData, &trainSize, &validSize, &testSize);

    int i, j, aux;

    OCLConfig* ocl = new OCLConfig();

    ocl->allocateBuffers(params, trainSize, validSize, testSize);
    ocl->setNDRages();
    ocl->setCompileFlags();
    //ocl->setProgramSource(params, &fullData);
    ocl->buildProgram(params, &fullData, "kernels\\kernel_split_data.cl");
    ocl->buildKernels();

    int* indexesData = new int[fullData.M];// save the index order given the data shuffle+folds generation
    for(aux = 0; aux < fullData.M; aux++){
        indexesData[aux] = aux;
    }
    int* indexesDataInFolds = new int[fullData.M - (fullData.M % KFOLDS)];// save the indexes given the folds generation

    for(i = 0; i < 3; i++) {
        for(aux = 0; aux < NUM_INDIV; aux++){
            seeds[aux] = i + 50;
        }

        shuffleData(&fullData, indexesData, &seeds[0]);
        Dataset* folds = generateFolds(&fullData, indexesData, indexesDataInFolds);


        for(j = 0; j < KFOLDS; j++){
            std::cout << "(" << i << " " << j << ")" << std::endl;

            int testIndex = j;
            int indexesFolds[KFOLDS];

            ///return a permutation of the possible indexes for training and validation
            getIndexes(indexesFolds, KFOLDS, testIndex, &seeds[0]);
            indexesFolds[KFOLDS-1] = testIndex;

            Dataset* trainingData = getSelectedDataset(folds, indexesFolds, 0, 6);
            Dataset* validationData = getSelectedDataset(folds, indexesFolds, 7, 8);
            Dataset* testData = getSelectedDataset(folds, indexesFolds, 9, 9);
            //std::cout << "(" << trainSize << " " << validSize << " " << testSize << ")" << std::endl;
            //std::cout << "(" << trainingData->M << " " << validationData->M << " " << testData->M << ")" << std::endl;

            ocl->transposeDatasets(trainingData, validationData, testData);

            timeManager.getStartTime(Evolucao_T);
            #if PARALLEL
                //Chromosome executionBest = PCGP_SeparateKernels(trainingData, validationData, params,  seeds);


                executionBest = PCGP(trainingData, validationData, params, ocl, seeds);
                std::cout << "Test execution: " << std::endl;

                std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;
                ocl->writeBestBuffer(&executionBest);

                ocl->finishCommandQueue();

                ocl->enqueueTestKernel();
                ocl->finishCommandQueue();

                ocl->readBestBuffer(&executionBest);
                ocl->finishCommandQueue();
                std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;

            #else
                executionBest = CGP(trainingData, validationData, params, seeds);
                std::cout << "Test execution: " << std::endl;
                std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;

                evaluateCircuit(&executionBest, testData);

                std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;
            #endif
            timeManager.getEndTime(Evolucao_T);
            std::cout << "Evol time  = " << timeManager.getElapsedTime(Evolucao_T) << std::endl;

            fprintf(f_CGP, "%d,\t%d,\t%.4f\n", i, j, executionBest.fitness);

        }


    }


    timeManager.getEndTime(Total_T);

    //printCircuit(&best, params);
    std::cout << "Best fitness  = " << executionBest.fitness << std::endl;

    std::cout << "Total time  = " << timeManager.getElapsedTime(Total_T) << std::endl;



    delete params;

    return 0;
}