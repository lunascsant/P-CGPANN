#include <iostream>

#include "cgp.h"
#include <fstream>
#include <map>
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

    std::string newSeed = argv[5];
    std::string geneNamesStr = argv[1];
    std::cout << geneNamesStr << std::endl;
    std::string argExe = argv[2];
    std::string argProblemName = argv[3];
    std::ifstream geneNamesFile(geneNamesStr);
    std::vector<std::string> geneNames;
    std::string gene;
    int numGenes = 0;

    while(std::getline (geneNamesFile, gene)) {
        geneNames.push_back(gene);
        numGenes++;
    }

    std::cout << geneNames.size() << std::endl;

    geneNamesFile.close();

    std::string currentGene = argv[4];

    std::vector<int> todasRedes;

    std::string datasetFile = currentGene + "_" + argProblemName + ".txt";



#if PARALLEL

    std::string unfeasiblesFile = "./executions_parallel/" + argExe + "/unfeasibles_" + argProblemName + ".txt";
    std::string rankedEdgesfile = "./executions_parallel/" + argExe + "/rankedEdges_" + argProblemName + ".csv";
    std::ofstream rankedEdges;
    std::ofstream unfeasibles;
    rankedEdges.open(rankedEdgesfile, std::ios_base::app);
    unfeasibles.open(unfeasiblesFile, std::ios_base::app);

    std::ofstream factivelFile;

    std::string nomeArquivo = currentGene + "_" + newSeed + "_" + argExe;
    std::string caminhoArquivo = "./executions_parallel/" + argExe + "/" + nomeArquivo + ".txt";
    factivelFile.open(caminhoArquivo, std::ios::out);
    if (!factivelFile) {
        std::cout << "Error file" << std::endl;
        exit(1);
    }

    std::string caminhoArquivoTime = "./time_counting/" + argExe + "/" + nomeArquivo + ".txt";
    FILE *f_CGP_time_parallel = fopen(caminhoArquivoTime.c_str(), "w");
#else

    std::string unfeasiblesFile = "./executions_sequential/" + argExe + "/unfeasibles_" + argProblemName + ".txt";
    std::string rankedEdgesfile = "./executions_sequential/" + argExe + "/rankedEdges_" + argProblemName + ".csv";
    std::ofstream rankedEdges;
    std::ofstream unfeasibles;
    rankedEdges.open(rankedEdgesfile, std::ios_base::app);
    unfeasibles.open(unfeasiblesFile, std::ios_base::app);

    std::ofstream factivelFile;

    std::string nomeArquivo = currentGene + "_" + newSeed + "_" + argExe;
    std::string caminhoArquivo = "./executions_sequential/" + argExe + "/" + nomeArquivo + ".txt";
    factivelFile.open(caminhoArquivo, std::ios::out);
    if (!factivelFile) {
        std::cout << "Error file" << std::endl;
        exit(1);
    }

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


    // O newSeed será o valor da SEED
    int* seeds;
    seeds = new int [ocl->maxLocalSize * NUM_INDIV];
    //seeds = new int [1];

    srand(atoi(argv[5]));

    /*random seeds used in parallel code*/
    for(int i = 0; i < ocl->maxLocalSize * NUM_INDIV; i++){
        seeds[i] = atoi(argv[5]);
    }



    std::vector<int> rede;
    int countUnfeasible;
    std::vector<int> rede_local;

    Dataset* trainingData = &fullData;

    ocl->transposeDatasets(trainingData);
    double timeIter = 0;
    double timeIterTotal = 0;
    double timeKernel = 0;
    timeManager.getStartTime(Evolucao_T);
#if PARALLEL
    Chromosome* executionBest = PCGP(trainingData, params, ocl, seeds, &timeIter, &timeKernel, factivelFile);

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        std::cout << "Fitness - exe " << i << " : " << executionBest[i].fitness << std::endl;
        if(executionBest[i].fitness == params->M)
            printFileFiveExe(&executionBest[i], params, factivelFile);
        else
            factivelFile << "Nao factivel\n\n";
    }

    countUnfeasible = 0;

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        if(executionBest[i].fitness != params->M) {
            countUnfeasible += 1;
            continue;
        }

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

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        std::cout << "Fitness - exe " << i << " : " <<executionBest[i].fitness << std::endl;
        if(executionBest[i].fitness == params->M)
            printFileFiveExe(&executionBest[i], params, factivelFile);
        else
            factivelFile << "Nao factivel\n\n";
    }

    countUnfeasible = 0;

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        if(executionBest[i].fitness != params->M) {
            countUnfeasible += 1;
            continue;
        }

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

#endif
    timeManager.getEndTime(Evolucao_T);
    timeIterTotal = timeManager.getElapsedTime(Evolucao_T);
    printf("Evol time: %f \n", timeIterTotal);

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

    for(int i = 0; i < geneNames.size(); i++) {
        if(counting.at(i) != 0) {
            rankedEdges << geneNames[i] << "\t" << currentGene << "\t" << counting.at(i) << "\n";
        }
    }

    if(countUnfeasible == NUM_EXECUTIONS) {
        unfeasibles << currentGene << "\n";
    }


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

    rankedEdges.close();

    return 0;
}