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
#define MAX_NODES_SIZE (3 * MAX_NODES)
#define MAX_OUTPUTS 2
#define MAX_ARITY 20


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


#define GPU 1
#define PARALLEL 1

typedef struct
{
    unsigned int N; //inputs
    unsigned int O; //outputs
    unsigned int M; //dataset size

    unsigned int NUM_FUNCTIONS;
    unsigned int* functionSet;
    //unsigned int* maxFunctionInputs;
    //unsigned int* inputVariablesSet;

    double weightRange;
    char** labels;
} Parameters;

typedef struct
{
    unsigned int N; //inputs
    unsigned int O; //outputs
    unsigned int M; //dataset size

    float** data;
    float** output;
} Dataset;

#endif //PCGP_CONSTANTS_H
