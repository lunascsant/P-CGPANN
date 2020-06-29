#define CONST_PI 3.14159265359


typedef struct
{
    unsigned int inputsEvaluated;
    unsigned int function;
    unsigned int maxInputs;
    unsigned int inputs[MAX_ARITY];
    double inputsWeight[MAX_ARITY];
    double output;
    int active;

} Node;

typedef struct
{
    Node nodes[MAX_NODES];
    unsigned int output[MAX_OUTPUTS];
    unsigned int numActiveNodes;
    double fitness;
    double fitnessValidation;
} Chromosome;

typedef struct {
    int topIndex;
    unsigned int info[MAX_NODES * MAX_ARITY];
} Stack;

typedef struct {
    int topIndex;
    double info[MAX_NODES * MAX_ARITY];
} ExStack;

void push(Stack* s  unsigned int info){
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES * MAX_ARITY){
        s->info[s->topIndex] = info;
    }
}

unsigned int pop(Stack* s){
    if(s->topIndex >= 0){
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    }
}

void pushEx(ExStack* s  double info) {
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES * MAX_ARITY){
        s->info[s->topIndex] = info;
    }
}

double popEx(ExStack* s) {
    if(s->topIndex >= 0){
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    }
}

int rand2(int *seed){
    int s  = *seed;
    s = ((unsigned int)(s * 16807) % 2147483647);//(int)(pown(2.0  31)-1));
    *seed = s;

    return s;
}

unsigned int randomInput(unsigned int index  int *seed) {
    return (rand2(seed) % (N + index));
}

unsigned int randomOutputIndex(int* seed){
    return (rand2(seed) % MAX_NODES);
}

unsigned int randomFunction(int *seed) {
    return (rand2(seed) % (NUM_FUNCTIONS));
}

double randomConnectionWeight(int *seed) {
    return (((double) rand2(seed) / (double) (2147483647) ) * 2 * WEIGTH_RANGE) - WEIGTH_RANGE;
}

int randomInterval(int inf_bound  int sup_bound  int *seed) {
    return rand2(seed) % (sup_bound - inf_bound + 1) + inf_bound;
}

double randomProb(int* seed){
    return (double)rand2(seed) / 2147483647;//pown(2.0  31);
}

unsigned int getFunctionInputs(unsigned int function){
    switch (function) {
        #ifdef ADD
        case ADD:
        #endif
        #ifdef SUB
        case SUB:
        #endif
        #ifdef MUL
        case MUL:
        #endif
        #ifdef DIV
        case DIV:
        #endif
        #ifdef AND
        case AND:
        #endif
        #ifdef OR
        case OR:
        #endif
        #ifdef XOR
        case XOR:
        #endif
        #ifdef NAND
        case NAND:
        #endif
        #ifdef NOR
        case NOR:
        #endif
        #ifdef XNOR
        case XNOR:
        #endif
        #ifdef SIG
        case SIG:
        #endif
        #ifdef GAUSS
        case GAUSS:
        #endif
        #ifdef STEP
        case STEP:
        #endif
        #ifdef SOFTSIGN
        case SOFTSIGN:
        #endif
        #ifdef TANH
        case TANH:
            return MAX_ARITY;
        #endif
        #ifdef RAND
        case RAND:
        #endif
        #ifdef PI
        case PI:
        #endif
        #ifdef ONE
        case ONE:
        #endif
        #ifdef ZERO
        case ZERO:
            return 0;
        #endif
        #ifdef ABS 
        case ABS:
        #endif
        #ifdef SQRT
        case SQRT:
        #endif
        #ifdef SQ
        case SQ:
        #endif
        #ifdef CUBE
        case CUBE:
        #endif
        #ifdef EXP
        case EXP:
        #endif
        #ifdef SIN
        case SIN:
        #endif
        #ifdef COS
        case COS:
        #endif
        #ifdef TAN
        case TAN:
        #endif
        #ifdef NOT
        case NOT:
        #endif
        #ifdef WIRE
        case WIRE:
            return 1;
        #endif
        #ifdef POW
        case POW:
            return 2;
        #endif
        default:
            break;
    }
}


