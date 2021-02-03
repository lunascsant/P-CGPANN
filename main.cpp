#include <iostream>

#include "cgp.h"
#include "utils.h"
#include "GPTime.h"
#include "OCLConfig.h"

int main(int argc, char** argv) {
    char* datasetFile = argv[1];

    std::string resultFile;
    std::string resultFileTime;
    std::string resultFileTimeIter;
    std::string resultFileTimeKernel;

    #if DEFAULT
        resultFile = "./results/cgpann_standard.txt";
        resultFileTime = "./results/cgpann_time_standard.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_standard.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_standard.txt";
    #elif COMPACT
        resultFile = "./results/cgpann_compact.txt";
        resultFileTime = "./results/cgpann_time_compact.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_compact.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_compact.txt";
    #elif IMAGE_R
        resultFile = "./results/cgpann_img_r.txt";
        resultFileTime = "./results/cgpann_time_img_r.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_img_r.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_img_r.txt";
    #elif IMAGE_RG
        resultFile = "./results/cgpann_img_rg.txt";
        resultFileTime = "./results/cgpann_time_img_rg.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_img_rg.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_img_rg.txt";
    #elif IMAGE_RGBA
        resultFile = "./results/cgpann_img_rgba.txt";
        resultFileTime = "./results/cgpann_time_img_rgba.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_img_rgba.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_img_rgba.txt";
    #elif COMPACT_R
        resultFile = "./results/cgpann_compact_img_r.txt";
        resultFileTime = "./results/cgpann_time_compact_img_r.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_compact_img_r.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_compact_img_r.txt";
    #elif COMPACT_RG
        resultFile = "./results/cgpann_compact_img_rg.txt";
        resultFileTime = "./results/cgpann_time_compact_img_rg.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_compact_img_rg.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_compact_img_rg.txt";
    #elif  COMPACT_RGBA
        resultFile = "./results/cgpann_compact_img_rgba.txt";
        resultFileTime = "./results/cgpann_time_compact_img_rgba.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_compact_img_rgba.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_compact_img_rgba.txt";
    #else
        resultFile = "./results/cgpann.txt";
        resultFileTime = "./results/cgpann_time.txt";
        resultFileTimeIter = "./results/cgpann_timeIter.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel.txt";
    #endif

    FILE *f_CGP = fopen(resultFile.c_str(), "w");
    FILE *f_CGP_time = fopen(resultFileTime.c_str(), "w");
    FILE *f_CGP_timeIter = fopen(resultFileTimeIter.c_str(), "w");
    FILE *f_CGP_timeKernel = fopen(resultFileTimeKernel.c_str(), "w");


    fprintf(f_CGP, "i,\tj,\taccuracy\n");
    //fprintf(f_CGP_time, "i,\tj,\ttime\n");
    //fprintf(f_CGP_timeKernel, "i,\tj,\ttime\n");

    GPTime timeManager(4);
    timeManager.getStartTime(Total_T);

    Parameters *params;
    params = new Parameters;

    Dataset fullData;
    readDataset(params, &fullData, datasetFile);

    int trainSize, validSize, testSize;
    calculateDatasetsSize(&fullData, &trainSize, &validSize, &testSize);

    int i, j, aux;

    /**OPENCL CONFIG */
    OCLConfig* ocl = new OCLConfig();
    ocl->allocateBuffers(params, trainSize, validSize, testSize);
    ocl->setNDRages();
    ocl->setCompileFlags();
    ocl->buildProgram(params, &fullData, "kernels\\kernel.cl");
    ocl->buildKernels();
#if IMAGE_R
    ocl->setupImageBuffers();
#elif IMAGE_RG
    ocl->setupImageBuffersHalf();
#elif IMAGE_RGBA
    ocl->setupImageBuffersQuarter();
#elif  COMPACT_R
    ocl->setupImageBuffersCompact();
#elif  COMPACT_RG
    ocl->setupImageBuffersHalfCompact();
#elif  COMPACT_RGBA
    ocl->setupImageBuffersQuarterCompact();
#endif
    /**OPENCL CONFIG */

    int* seeds;
    seeds = new int [ocl->maxLocalSize * NUM_INDIV];
    srand(SEED);

    /*random seeds used in parallel code*/
    for(i = 0; i < ocl->maxLocalSize * NUM_INDIV; i++){
        seeds[i] = rand();
    }


    int* indexesData = new int[fullData.M];// save the index order given the data shuffle+folds generation
    for(aux = 0; aux < fullData.M; aux++){
        indexesData[aux] = aux;
    }
    int* indexesDataInFolds = new int[fullData.M - (fullData.M % KFOLDS)];// save the indexes given the folds generation

    for(i = 0; i < 3; i++) {
        for(aux = 0; aux < ocl->maxLocalSize * NUM_INDIV; aux++){
            seeds[aux] = aux + 55;
        }

        shuffleData(&fullData, indexesData, &seeds[0]);
        Dataset* folds = generateFolds(&fullData, indexesData, indexesDataInFolds);
        int id;
        //#pragma omp parallel for default(none), private(j, id), shared(i, params, folds, f_CGP, timeManager, seeds, ocl), schedule(dynamic), num_threads(10)
        for(j = 0; j < KFOLDS; j++){
            printf("( %d %d )\n", i, j);
            for(aux = 0; aux < ocl->maxLocalSize * NUM_INDIV; aux++){
                seeds[aux] = aux + 55 + i + j;
            }
            //std::cout << "(" << i << " " << j << ")" << std::endl;

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
            double timeIter = 0;
            double timeIterTotal = 0;
            double timeKernel = 0;
            timeManager.getStartTime(Evolucao_T);
            #if PARALLEL
                Chromosome executionBest = PCGP(trainingData, validationData, params, ocl, seeds, &timeIter, &timeKernel);

                std::cout << "Test execution: " << std::endl;

                std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;
                ocl->writeBestBuffer(&executionBest);

                ocl->finishCommandQueue();

                ocl->enqueueTestKernel();
                ocl->finishCommandQueue();

                //ocl->readBestBuffer(&executionBest);
                ocl->readFitnessBuffer();

                executionBest.fitness = ocl->fitness[0];

                ocl->finishCommandQueue();
                std::cout << executionBest.fitness << std::endl;

            #else
                Chromosome executionBest = CGP(trainingData, validationData, params, seeds, &timeIter, &timeKernel);
                //std::cout << "Test execution: " << std::endl;
                std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;

                evaluateCircuit(&executionBest, testData);
                printf("Test execution: %f ", executionBest.fitness);

                //std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;
            #endif
            timeManager.getEndTime(Evolucao_T);
            timeIterTotal = timeManager.getElapsedTime(Evolucao_T);
            printf("Evol time: %f \n", timeIterTotal);

            //std::cout << "Evol time  = " << timeManager.getElapsedTime(Evolucao_T) << std::endl;

            fprintf(f_CGP, "%d,\t%d,\t%.4f\n", i, j, executionBest.fitness);
            fprintf(f_CGP_time, "%d;\t%d;\t%.4f\n", i, j, timeIter);
            fprintf(f_CGP_timeIter, "%d;\t%d;\t%.4f\n", i, j, timeIterTotal);
            fprintf(f_CGP_timeKernel, "%d;\t%d;\t%.4f\n", i, j, timeKernel);

        }


    }


    timeManager.getEndTime(Total_T);

    //printCircuit(&best, params);
    //std::cout << "Best fitness  = " << executionBest.fitness << std::endl;

    std::cout << "Total time  = " << timeManager.getElapsedTime(Total_T) << std::endl;



    delete params;

    return 0;
}