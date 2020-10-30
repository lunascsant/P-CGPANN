//
// Created by bruno on 04/02/2020.
//

#ifndef PCGP_CONSTANTS_H
#define PCGP_CONSTANTS_H

#define SEED 10

/** Available node functions */
#define AND 10
#define OR 11
#define XOR 12
#define NOT 14
#define NAND 15
#define NOR 16
#define XNOR 17

#define ADD 18
#define SUB 19
#define MUL 20
#define DIV 21
#define ABS 22
#define SQRT 23
#define SQ 24
#define CUBE 25
#define POW 26

#define EXP 27
#define SIN 28
#define COS 29
#define TAN 30
#define SIG 31
#define GAUSS 32
#define STEP 33
#define SOFTSIGN 34
#define TANH 35
#define RAND 36
#define PI 37
#define ONE 38
#define ZERO 39
#define WIRE 40

#define CONST_PI 3.14159265359

/** Graph parameters */
#define MAX_NODES 500
#define MAX_OUTPUTS 4
#define MAX_ARITY 20

#define TESTVAR MAX_OUTPUTS/2 + MAX_OUTPUTS%2
/** Genetic parameters */
#ifndef NUM_INDIV
#define NUM_INDIV (5)
#endif // NUM_INDIV

#ifndef PROB_CROSS
#define PROB_CROSS 0.9
#endif // PROB_CROSS

#ifndef PROB_MUT
#define PROB_MUT 0.05
#endif // PROB_MUT


#define NUM_GENERATIONS 50000
#define NUM_EVALUATIONS 2.40e+007

#define RANDOM_WEIGHT 0


#define PARALLEL    1

#define DEFAULT      0
#define COMPACT      1
#define IMAGE_R      0
#define IMAGE_RG     0
#define IMAGE_RGBA   0
#define COMPACT_R    0
#define COMPACT_RG   0
#define COMPACT_RGBA 0


#define KFOLDS 40

#define TRAIN_FOLDS 30 //7 * 4
#define VALID_FOLDS 9 //2 * 4
#define TEST_FOLDS 1




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
    unsigned int function;
    unsigned int maxInputs;
    unsigned int inputs[MAX_ARITY];
    float inputsWeight[MAX_ARITY];
    int active;
} Node;

typedef struct
{
    Node nodes[MAX_NODES];
    unsigned int output[MAX_OUTPUTS];
    unsigned int activeNodes[MAX_NODES];
    unsigned int numActiveNodes;
    float fitness;
    float fitnessValidation;
} Chromosome;

typedef struct
{
    unsigned int function_inputs_active;
    unsigned int inputs[MAX_ARITY/2];
    float inputsWeight[MAX_ARITY];
} CompactNode;

typedef struct
{
    CompactNode nodes[MAX_NODES];
    unsigned int output[MAX_OUTPUTS];
    unsigned int activeNodes[MAX_NODES/2];
    unsigned int numActiveNodes;
} CompactChromosome;


typedef struct
{
    unsigned int function;
    unsigned int maxInputs;
    unsigned int inputs[MAX_ARITY];
    float inputsWeight[MAX_ARITY];
    unsigned int originalIndex;

} ActiveNode;

typedef struct
{
    ActiveNode nodes[MAX_NODES];
    unsigned int output[MAX_OUTPUTS];
    unsigned int numActiveNodes;
} ActiveChromosome;


typedef struct
{
    unsigned int N; //inputs
    unsigned int O; //outputs
    unsigned int M; //dataset size

    unsigned int NUM_FUNCTIONS;
    unsigned int* functionSet;
    //unsigned int* maxFunctionInputs;
    //unsigned int* inputVariablesSet;

    float weightRange;
    char** labels;
} Parameters;

typedef struct
{
    /** Number of inputs */
    unsigned int N;
    /** Number of outputs */
    unsigned int O;
    /** Number of entries */
    unsigned int M;

    float** data;
    float** output;
} Dataset;

#endif //PCGP_CONSTANTS_H
