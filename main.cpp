#include <iostream>

#include "cgp.h"
#include <fstream>
#include "utils.h"
#include "GPTime.h"
#include "OCLConfig.h"



int main(int argc, char** argv) {
    char* datasetFile = argv[1];

    #if PARALLEL
        std::string gene = argv[1];
        if (gene[10] == '0') {
            gene = gene.substr(0, 21);
        } else {
            gene = gene.substr(0, 20);
        }

        std::string argSeed = argv[2];
        std::string argExe = argv[3];
        std::string nomeArquivo = gene + "_" + argSeed + "_" + argExe;
        std::string caminhoArquivo = "./executions_parallel/" + argExe + "/" + nomeArquivo;
        std::string caminhoArquivoTime = "./time_counting/" + argExe + "/" + nomeArquivo;
        std::string caminhoArquivoTimeGeral = caminhoArquivoTime + "_geral.txt";
        FILE *f_CGP_time_parallel_5_executions = fopen(caminhoArquivoTimeGeral.c_str(), "w");
    #else

        std::string gene = argv[1];
        if (gene[10] == '0') {
            gene = gene.substr(0, 21);
        } else {
            gene = gene.substr(0, 20);
        }
        std::string argSeed = argv[2];
        std::string argExe = argv[3];
        std::string nomeArquivo = gene + "_" + argSeed + "_" + argExe;
        std::string caminhoArquivo = "./executions_sequential/" + argExe + "/" + nomeArquivo;


        /*std::string resultFile;
        std::string resultFileTime;
        std::string resultFileTimeIter;
        std::string resultFileTimeKernel;*/
        //std::string caminhoArquivoTime = "./time_counting_sequential/" + argExe + "/" + nomeArquivo + ".txt";
        std::string caminhoArquivoTime = "./time_counting_sequential/" + nomeArquivo;
        std::string caminhoArquivoTimeGeral = caminhoArquivoTime + "_geral.txt";
        FILE *f_CGP_time_sequential_5_executions = fopen(caminhoArquivoTimeGeral.c_str(), "w");
    #endif

    #if DEFAULT
        /*resultFile = "./results/cgpann_standard.txt";
        resultFileTime = "./results/cgpann_time_standard.txt";
        resultFileTimeIter = "./results/cgpann_timeIter_standard.txt";
        resultFileTimeKernel = "./results/cgpann_timeKernel_standard.txt";*/
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

    /*FILE *f_CGP = fopen(resultFile.c_str(), "w");
    FILE *f_CGP_time = fopen(resultFileTime.c_str(), "w");
    FILE *f_CGP_timeIter = fopen(resultFileTimeIter.c_str(), "w");
    FILE *f_CGP_timeKernel = fopen(resultFileTimeKernel.c_str(), "w");*/


    // fprintf(f_CGP, "i,\tj,\taccuracy\n");


    GPTime timeManager(4);
    timeManager.getStartTime(Total_T);

    Parameters *params;
    params = new Parameters;

    Dataset fullData;
    readDataset(params, &fullData, datasetFile);
    //std::cout << "-----------------PRINT DATASET-------------------" << std::endl;
    //printDataset(&fullData);
    //std::cout << "-----------------PRINT DATASET-------------------" << std::endl;

    int trainSize, validSize, testSize;
    calculateDatasetsSize(&fullData, &trainSize, &validSize, &testSize);

    int i, aux;

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


    // O argv[2] será o valor da SEED
    int* seeds;
    seeds = new int [ocl->maxLocalSize * NUM_INDIV];
    std::cout << "SEED LIDA " << atoi(argv[2]) << std::endl;
    srand(atoi(argv[2]));

    /*random seeds used in parallel code*/
    for(i = 0; i < ocl->maxLocalSize * NUM_INDIV; i++){
        seeds[i] = atoi(argv[2]);
    }

    /*std::cout << "SEED vetor " << std::endl;
    for(i = 0; i < ocl->maxLocalSize * NUM_INDIV; i++){
        std::cout << seeds[i] << " ";
    }
    std::cout << std::endl;*/


    int* indexesData = new int[fullData.M];// save the index order given the data shuffle+folds generation
    for(aux = 0; aux < fullData.M; aux++){
        indexesData[aux] = aux;
    }

    int* indexesDataInFolds = new int[fullData.M - (fullData.M % KFOLDS)];// save the indexes given the folds generation

    for(i = 0; i < 1; i++) {
        /*for(aux = 0; aux < ocl->maxLocalSize * NUM_INDIV; aux++){
            seeds[aux] = aux + 55;
        }*/

        //shuffleData(&fullData, indexesData, &seeds[0]);
        //Dataset* folds = generateFolds(&fullData, indexesData, indexesDataInFolds);
        int id;
        //#pragma omp parallel for default(none), private(j, id), shared(i, params, folds, f_CGP, timeManager, seeds, ocl), schedule(dynamic), num_threads(10)
        //for(j = 0; j < KFOLDS; j++){
            printf("( %d )\n", i);
            /*for(aux = 0; aux < ocl->maxLocalSize * NUM_INDIV; aux++){
                seeds[aux] = aux + 55 + i;
            }*/
            //std::cout << "(" << i << " " << j << ")" << std::endl;

            int testIndex = 0;
            int indexesFolds[KFOLDS];

            ///return a permutation of the possible indexes for training and validation
            //getIndexes(indexesFolds, KFOLDS, testIndex, &seeds[0]);
            //indexesFolds[KFOLDS-1] = testIndex;

            //Dataset* trainingData = getSelectedDataset(folds, indexesFolds, 0, 0);
            Dataset* trainingData = &fullData;
            //Dataset* validationData = getSelectedDataset(folds, indexesFolds, 7, 8);
            //Dataset* testData = getSelectedDataset(folds, indexesFolds, 9, 9);
            //std::cout << "(" << trainSize << " " << validSize << " " << testSize << ")" << std::endl;
            //std::cout << "(" << trainingData->M << " " << validationData->M << " " << testData->M << ")" << std::endl;

            ocl->transposeDatasets(trainingData);
            double timeIter = 0;
            double timeIterTotal = 0;
            double timeKernel = 0;
            timeManager.getStartTime(Evolucao_T);
            #if PARALLEL
                PCGP(trainingData, params, ocl, seeds, &timeIter, &timeKernel, caminhoArquivo, caminhoArquivoTime);

                //std::cout << "Test execution: " << std::endl;

                //std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;
                //std::cout << "Fitness do melhor na main: " << executionBest.fitness << std::endl;
                //ocl->writeBestBuffer(&executionBest);

                //ocl->finishCommandQueue();

                //ocl->enqueueTestKernel();
                //ocl->finishCommandQueue();

                //ocl->readBestBuffer(&executionBest);
                //ocl->readFitnessBuffer();

                //executionBest.fitness = ocl->fitness[0];

                //ocl->finishCommandQueue();
                //std::cout << executionBest.fitness << std::endl;

            #else
                CGP(trainingData, params, seeds, &timeIter, &timeKernel, caminhoArquivo);
                //std::cout << "Test execution: " << std::endl;
                //std::cout << "Melhor na main: " << executionBest.fitness << std::endl;

                /*evaluateCircuit(&executionBest, testData);
                printf("Test execution: %f ", executionBest.fitness);*/

                //std::cout << executionBest.fitness << " " << executionBest.fitnessValidation << std::endl;
            #endif
            timeManager.getEndTime(Evolucao_T);
            timeIterTotal = timeManager.getElapsedTime(Evolucao_T);
            printf("Evol time: %f \n", timeIterTotal);

            //std::cout << "Evol time  = " << timeManager.getElapsedTime(Evolucao_T) << std::endl;

            /*fprintf(f_CGP, "%d,\t%d,\t%.4f\n", i, 0, executionBest.fitness);
            fprintf(f_CGP_time, "%d;\t%d;\t%.4f\n", i, 0, timeIter);
            fprintf(f_CGP_timeIter, "%d;\t%d;\t%.4f\n", i, 0, timeIterTotal);
            fprintf(f_CGP_timeKernel, "%d;\t%d;\t%.4f\n", i, 0, timeKernel);*/

            #if PARALLEL
                fprintf(f_CGP_time_parallel_5_executions, "timeIter: \t%.4f\n", timeIter);
                fprintf(f_CGP_time_parallel_5_executions, "timeIterTotal: \t%.4f\n", timeIterTotal);
                fprintf(f_CGP_time_parallel_5_executions, "timeKernel: \t%.4f\n", timeKernel);
            #else
                //fprintf(f_CGP_time_sequential_5_executions, "Fitness best: \t%.4f\n", executionBest.fitness);
                fprintf(f_CGP_time_sequential_5_executions, "timeIter: \t%.4f\n", timeIter);
                fprintf(f_CGP_time_sequential_5_executions, "timeIterTotal: \t%.4f\n", timeIterTotal);
                fprintf(f_CGP_time_sequential_5_executions, "timeKernel: \t%.4f\n", timeKernel);
            #endif


        //}


    }


    timeManager.getEndTime(Total_T);

    //printCircuit(&best, params);
    //std::cout << "Best fitness  = " << executionBest.fitness << std::endl;
    #if PARALLEL
        fprintf(f_CGP_time_parallel_5_executions, "Total time: \t%.4f\n", timeManager.getElapsedTime(Total_T));
        fprintf(f_CGP_time_parallel_5_executions, "\n");
    #else
        fprintf(f_CGP_time_sequential_5_executions, "Total time: \t%.4f\n", timeManager.getElapsedTime(Total_T));
        fprintf(f_CGP_time_sequential_5_executions, "\n");
    #endif

    std::cout << "Total time  = " << timeManager.getElapsedTime(Total_T) << std::endl;


    //factivelFile.close();
    delete params;

    return 0;
}