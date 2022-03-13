#include <iostream>

#include "cgp.h"
#include <fstream>
#include "utils.h"
#include "GPTime.h"
#include "OCLConfig.h"

void printFileFiveExe(Chromosome *c, Parameters *p, std::ofstream& factivel_file) {
    for(int i = 0; i < MAX_NODES; i++) {
        if (c->nodes[i].active) {
            factivel_file << "Node" << i + p->N << " " << c->nodes[i].inputs[0]
            << " " << c->nodes[i].inputs[1] << " " <<  c->nodes[i].function << "\n";
        }
    }
    factivel_file << "Output " << c->output[0] + p->N << "\n";
    factivel_file << "\n";
    factivel_file << "\n";
}

int main(int argc, char** argv) {
    char* datasetFile = argv[1];

    std::string genenames = argv[4];

    #if PARALLEL
        std::ofstream factivelFile;
        std::string gene = argv[1];
        std::string edgesDir;
        if (gene[10] == '0') {
            gene = gene.substr(0, 21);
            edgesDir = gene.substr(0, 11);
        } else {
            gene = gene.substr(0, 20);
            edgesDir = gene.substr(0, 10);
        }
        std::cout << edgesDir << std::endl;
        std::string argSeed = argv[2];
        std::string argExe = argv[3];
        std::string argPt = argv[5];
        std::string argCurrentGene = argv[6];
        std::string nomeArquivo = gene + "_" + argSeed + "_" + argExe;
        std::string caminhoArquivo = "./executions_parallel/" + argExe + "/" + nomeArquivo + ".txt";
        std::string rankedEdgesfile = "./executions_parallel/" + argExe + "/" + edgesDir + "/rankedEdges_" + argPt + "_" + argCurrentGene + ".csv";
        factivelFile.open(caminhoArquivo, std::ios::out);
        if (!factivelFile) {
            std::cout << "Error file" << std::endl;
            exit(1);
        }

        /*std::string resultFile;
        std::string resultFileTime;
        std::string resultFileTimeIter;
        std::string resultFileTimeKernel;*/
        std::string caminhoArquivoTime = "./time_counting/" + argExe + "/" + nomeArquivo + ".txt";
        FILE *f_CGP_time_parallel = fopen(caminhoArquivoTime.c_str(), "w");
    #else

        std::ofstream factivelFile;
        std::string gene = argv[1];
        std::string edgesDir;
        if (gene[10] == '0') {
            gene = gene.substr(0, 21);
            edgesDir = gene.substr(0, 11);
        } else {
            gene = gene.substr(0, 20);
            edgesDir = gene.substr(0, 10);
        }
        std::string argSeed = argv[2];
        std::string argExe = argv[3];
        std::string argPt = argv[5];
        std::string argCurrentGene = argv[6];
        std::string nomeArquivo = gene + "_" + argSeed + "_" + argExe;
        std::string caminhoArquivo = "./executions_sequential/" + argExe + "/" + nomeArquivo + ".txt";
        std::string rankedEdgesfile = "./executions_sequential/" + argExe + "/" + edgesDir + "/rankedEdges_" + argPt + "_" + argCurrentGene + ".csv";
        factivelFile.open(caminhoArquivo, std::ios::out);
        if (!factivelFile) {
            std::cout << "Error file" << std::endl;
            exit(1);
        }

        /*std::string resultFile;
        std::string resultFileTime;
        std::string resultFileTimeIter;
        std::string resultFileTimeKernel;*/
        std::string caminhoArquivoTime = "./time_counting_sequential/" + argExe + "/" + nomeArquivo + ".txt";
        FILE *f_CGP_time_sequential = fopen(caminhoArquivoTime.c_str(), "w");
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
    //seeds = new int [1];

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

    std::vector<int> rede;
    for(i = 0; i < 1; i++) {
        std::vector<int> rede_local;
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
            Chromosome* executionBest = PCGP(trainingData, params, ocl, seeds, &timeIter, &timeKernel, factivelFile);

                for(int i = 0; i < NUM_EXECUTIONS; i++) {
                    std::cout << "Fitness - exe " << i << " : " <<executionBest[i].fitness << std::endl;
                    printFileFiveExe(&executionBest[i], params, factivelFile);
                }

                for(int i = 0; i < NUM_EXECUTIONS; i++) {
                    for(int j = 0; j < MAX_NODES; j++){
                        if(executionBest[i].nodes[j].active == 1){
                            for(int k = 0; k < MAX_ARITY; k++){
                                if(executionBest[i].nodes[j].inputs[k] < trainingData->N){
                                    //std::cout << "Input: " << k << " " << executionBest.nodes[j].inputs[k] << std::endl;
                                    auto search = find(rede_local.begin(), rede_local.end(), executionBest[i].nodes[j].inputs[k]);
                                    if(search == rede_local.end()){
                                        rede_local.push_back(executionBest[i].nodes[j].inputs[k]);
                                    }
                                }
                            }

                        }
                    }

                    for(int & j : rede_local){
                        rede.push_back(j);
                    }

                    rede_local.clear();
                }



            #else
                Chromosome* executionBest = CGP(trainingData, params, seeds, &timeIter, &timeKernel, factivelFile);
                //std::cout << "Test execution: " << std::endl;
                //std::cout << "Melhor na main: " << executionBest.fitness << std::endl;

                for(int i = 0; i < NUM_EXECUTIONS; i++) {
                    std::cout << "Fitness - exe " << i << " : " <<executionBest[i].fitness << std::endl;
                    printFileFiveExe(&executionBest[i], params, factivelFile);
                }

                for(int i = 0; i < NUM_EXECUTIONS; i++) {
                    for(int j = 0; j < MAX_NODES; j++){
                        if(executionBest[i].nodes[j].active == 1){
                            for(int k = 0; k < MAX_ARITY; k++){
                                if(executionBest[i].nodes[j].inputs[k] < trainingData->N){
                                    //std::cout << "Input: " << k << " " << executionBest.nodes[j].inputs[k] << std::endl;
                                    auto search = find(rede_local.begin(), rede_local.end(), executionBest[i].nodes[j].inputs[k]);
                                    if(search == rede_local.end()){
                                        rede_local.push_back(executionBest[i].nodes[j].inputs[k]);
                                    }
                                }
                            }

                        }
                    }

                    for(int & j : rede_local){
                        rede.push_back(j);
                    }

                    rede_local.clear();
                }

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
                /*fprintf(f_CGP_time_parallel, "Fitness best: \t%.4f\n", executionBest.fitness);*/
                fprintf(f_CGP_time_parallel, "timeIter: \t%.4f\n", timeIter);
                fprintf(f_CGP_time_parallel, "timeIterTotal: \t%.4f\n", timeIterTotal);
                fprintf(f_CGP_time_parallel, "timeKernel: \t%.4f\n\n", timeKernel);
            #else
                fprintf(f_CGP_time_sequential, "Fitness best: \t%.4f\n", executionBest->fitness);
                fprintf(f_CGP_time_sequential, "timeIter: \t%.4f\n", timeIter);
                fprintf(f_CGP_time_sequential, "timeIterTotal: \t%.4f\n", timeIterTotal);
                fprintf(f_CGP_time_sequential, "timeKernel: \t%.4f\n\n", timeKernel);
            #endif


        //}


    }


    timeManager.getEndTime(Total_T);

    std::vector<float> counting;

    for(int i = 0; i < fullData.N; i++){
        float counted = std::count(rede.begin(), rede.end(), i);
        counting.push_back(counted/NUM_EXECUTIONS);
    }

    std::cout << "xxxxxx Contagem xxxxxx" << std::endl;

    for(int i = 0; i < counting.size(); i++){
        std::cout << counting.at(i) << " ";
    }

    std::cout << std::endl;

    std::vector<std::string> geneNames = {};
    std::string names;
    std::ifstream FileGeneNames(genenames);
    int linha = 0;
    while(std::getline (FileGeneNames, names)) {
        geneNames.push_back(names);
        linha++;
    }
    FileGeneNames.close();

    std::ofstream rankedEdges;
    rankedEdges.open(rankedEdgesfile);
    for(int i = 0; i < geneNames.size(); i++) {
        if(counting.at(i) != 0) {
            rankedEdges << geneNames[i] << "\t" << argCurrentGene << "\t" << counting.at(i) << "\n";
        }
    }
    rankedEdges.close();

    #if PARALLEL
        fprintf(f_CGP_time_parallel, "\n");
        fprintf(f_CGP_time_parallel, "Total time: \t%.4f\n", timeManager.getElapsedTime(Total_T));
        fprintf(f_CGP_time_parallel, "\n");
    #else
        fprintf(f_CGP_time_sequential, "\n");
        fprintf(f_CGP_time_sequential, "Total time: \t%.4f\n", timeManager.getElapsedTime(Total_T));
        fprintf(f_CGP_time_sequential, "\n");
    #endif

    std::cout << "Total time  = " << timeManager.getElapsedTime(Total_T) << std::endl;


    factivelFile.close();
    delete params;

    return 0;
}