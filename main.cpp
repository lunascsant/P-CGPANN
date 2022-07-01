#include <iostream>
#include "string.h"
#include "cgp.h"
#include <fstream>
#include <map>
#include "utils.h"
#include "GPTime.h"
#include "OCLConfig.h"
#ifdef WIN32
#include "io.h"
#else
void mkdir(const char *path)
{
    char cmd[256];
    sprintf(cmd, "mkdir -p %s", path);
    system(cmd);
}
#endif


void printFeasibleFile(Chromosome *c, Parameters *p, std::ofstream& factivel_file) {
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

// entrada na linha de comando: tabela_verdade exe_n seed
int main(int argc, char** argv) {

    std::string dataset = argv[1];
    std::string datasetFile = dataset + ".txt";
    std::string argExe = argv[2];
    std::string newSeed = argv[3];

    std::string exe_1 = "exe_1";

#if PARALLEL

    if(strcmp(argExe.c_str(), exe_1.c_str())) {
        mkdir("./executions_parallel");
        std::string path_exe = "./executions_parallel/" + argExe;
        mkdir(path_exe.c_str());
    } else {
        std::string path_exe = "./executions_parallel/" + argExe;
        mkdir(path_exe.c_str());
    }

    std::string unfeasiblesFile = "./executions_parallel/" + argExe + "/unfeasibles.txt";
    std::ofstream unfeasibles;
    unfeasibles.open(unfeasiblesFile, std::ios_base::app);

    std::ofstream factivelFile;

    std::string nameFile = dataset + "_" + newSeed + "_" + argExe;
    std::string pathFile = "./executions_parallel/" + argExe + "/" + nameFile + ".txt";
    factivelFile.open(pathFile, std::ios::out);
    if (!factivelFile) {
        std::cout << "Error file" << std::endl;
        exit(1);
    }

    if(strcmp(argExe.c_str(), exe_1.c_str())) {
        mkdir("./time_counting_par");
        std::string path_exe = "./time_counting_par/" + argExe;
        mkdir(path_exe.c_str());
    } else {
        std::string path_exe = "./time_counting_par/" + argExe;
        mkdir(path_exe.c_str());
    }

    std::string pathFileTime = "./time_counting_par/" + argExe + "/" + nameFile + ".txt";
    FILE *f_CGP_time_parallel = fopen(pathFileTime.c_str(), "w");
#else

    if(strcmp(argExe.c_str(), exe_1.c_str())) {
        mkdir("./executions_sequential");
        std::string path_exe = "./executions_sequential/" + argExe;
        mkdir(path_exe.c_str());
    } else {
        std::string path_exe = "./executions_sequential/" + argExe;
        mkdir(path_exe.c_str());
    }

    std::string unfeasiblesFile = "./executions_sequential/" + argExe + "/unfeasibles.txt";
    std::ofstream unfeasibles;
    unfeasibles.open(unfeasiblesFile, std::ios_base::app);

    std::ofstream factivelFile;

    std::string nameFile = datasetFile + "_" + newSeed + "_" + argExe;
    std::string pathFile = "./executions_sequential/" + argExe + "/" + nameFile + ".txt";
    factivelFile.open(pathFile, std::ios::out);
    if (!factivelFile) {
        std::cout << "Error file" << std::endl;
        exit(1);
    }

    if(strcmp(argExe.c_str(), exe_1.c_str())) {
        mkdir("./time_counting_seq");
        std::string path_exe = "./time_counting_seq/" + argExe;
        mkdir(path_exe.c_str());
    } else {
        std::string path_exe = "./time_counting_seq/" + argExe;
        mkdir(path_exe.c_str());
    }

    std::string pathFileTime = "./time_counting_seq/" + argExe + "/" + nameFile + ".txt";
    FILE *f_CGP_time_sequential = fopen(pathFileTime.c_str(), "w");
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

    GPTime timeManager(4);
    timeManager.getStartTime(Total_T);

    Parameters *params;
    params = new Parameters;

    Dataset fullData;
    readDataset(params, &fullData, datasetFile);
    std::cout << "-----------------PRINT DATASET-------------------" << std::endl;
    printDataset(&fullData);
    std::cout << "-----------------PRINT DATASET-------------------" << std::endl;

    int trainSize, validSize, testSize;
    calculateDatasetsSize(&fullData, &trainSize, &validSize, &testSize);

#ifdef WIN32
    std::string kernel_str = "kernels\\kernel.cl";
#else
    std::string kernel_str = "kernels/kernel.cl";
#endif
    /**OPENCL CONFIG */
    OCLConfig* ocl = new OCLConfig();
    ocl->allocateBuffers(params, trainSize, validSize, testSize);
    ocl->setNDRages();
    ocl->setCompileFlags();
    ocl->buildProgram(params, &fullData, kernel_str.c_str());
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

    srand(atoi(argv[3]));

    /*random seeds used in parallel code*/
    for(int i = 0; i < ocl->maxLocalSize * NUM_INDIV; i++){
        seeds[i] = atoi(argv[3]);
    }

    Dataset* trainingData = &fullData;

    ocl->transposeDatasets(trainingData);
    double timeIter = 0;
    double timeIterTotal = 0;
    double timeKernel = 0;
    timeManager.getStartTime(Evolucao_T);
#if PARALLEL
    Chromosome executionBest = PCGP(trainingData, params, ocl, seeds, &timeIter, &timeKernel, factivelFile);

    std::cout << "Fitness: " << executionBest.fitness << std::endl;
    if(executionBest.fitness == (params->M * params->O))
        printFeasibleFile(&executionBest, params, factivelFile);
    else {
        factivelFile << "Nao factivel\n\n";
        printFeasibleFile(&executionBest, params, factivelFile);
        unfeasibles << datasetFile << "\n";
    }

#else
    Chromosome executionBest = CGP(trainingData, params, seeds, &timeIter, &timeKernel, factivelFile);

    std::cout << "Fitness: " << executionBest.fitness << std::endl;
    if(executionBest.fitness == params->M)
        printFeasibleFile(&executionBest, params, factivelFile);
    else {
        factivelFile << "Nao factivel\n\n";
        unfeasibles << datasetFile << "\n";
    }

#endif

    timeManager.getEndTime(Evolucao_T);
    timeIterTotal = timeManager.getElapsedTime(Evolucao_T);
    printf("Evol time: %f \n", timeIterTotal);

#if PARALLEL
    fprintf(f_CGP_time_parallel, "Fitness best: \t%.4f\n", executionBest.fitness);
    fprintf(f_CGP_time_parallel, "Transistors best: \t%d\n", executionBest.numTransistors);
    fprintf(f_CGP_time_parallel, "timeIter: \t%.4f\n", timeIter);
    fprintf(f_CGP_time_parallel, "timeIterTotal: \t%.4f\n", timeIterTotal);
    fprintf(f_CGP_time_parallel, "timeKernel: \t%.4f\n\n", timeKernel);
#else
    fprintf(f_CGP_time_sequential, "Fitness best: \t%.4f\n", executionBest.fitness);
    fprintf(f_CGP_time_sequential, "Transistors best: \t%d\n", executionBest.numTransistors);
    fprintf(f_CGP_time_sequential, "timeIter: \t%.4f\n", timeIter);
    fprintf(f_CGP_time_sequential, "timeIterTotal: \t%.4f\n", timeIterTotal);
    fprintf(f_CGP_time_sequential, "timeKernel: \t%.4f\n\n", timeKernel);
#endif

    timeManager.getEndTime(Total_T);

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
    unfeasibles.close();
    return 0;
}