#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "float precision floating point not supported by OpenCL implementation."
#endif

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                          CLK_ADDRESS_NONE |
                          CLK_FILTER_NEAREST; //Don't interpolate

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
    int activeNodes[MAX_NODES];
    unsigned int numActiveNodes;
    float fitness;
    float fitnessValidation;
} Chromosome;

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

typedef struct {
    int topIndex;
    unsigned int info[MAX_NODES * MAX_ARITY];
} Stack;

typedef struct {
    int topIndex;
    float info[MAX_NODES * MAX_ARITY];
} ExStack;

typedef struct {
    int topIndex;
    float info[MAX_ARITY];
} ExStackLinear;

void push(Stack* s, unsigned int info){
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

void pushEx(ExStack* s, float info) {
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES * MAX_ARITY){
        s->info[s->topIndex] = info;
    }
}

float popEx(ExStack* s) {
    if(s->topIndex >= 0){
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    }
}


void pushExLinear(ExStackLinear* s, float info) {
    (s->topIndex)++;
    s->info[s->topIndex] = info;
}

float popExLinear(ExStackLinear* s) {
    (s->topIndex)--;
    return s->info[(s->topIndex) + 1];
}

int rand2(int *seed){
    int s  = *seed;
    s = ((unsigned int)(s * 16807) % 2147483647);//(int)(pown(2.0, 31)-1));
    *seed = s;

    return s;
}

unsigned int randomInput(unsigned int index, int *seed) {
    return (rand2(seed) % (N + index));
}

unsigned int randomOutputIndex(int* seed){
    return (rand2(seed) % MAX_NODES);
}

unsigned int randomFunction(int *seed) {
    return (rand2(seed) % (NUM_FUNCTIONS));
}

float randomConnectionWeight(int *seed) {
    return (((float) rand2(seed) / (float) (2147483647) ) * 2 * WEIGTH_RANGE) - WEIGTH_RANGE;
}

int randomInterval(int inf_bound, int sup_bound, int *seed) {
    return rand2(seed) % (sup_bound - inf_bound + 1) + inf_bound;
}

float randomProb(int* seed){
    return (float)rand2(seed) / 2147483647;//pown(2.0, 31);
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

void activateNodes(__global Chromosome* c){

    int i, j;
    int alreadyEvaluated[MAX_NODES];
    for(i = 0; i < MAX_NODES; i++) {
        alreadyEvaluated[i] = -1;
        c->activeNodes[i] = MAX_NODES + 1;
        c->nodes[i].active = 0;
    }

    c->numActiveNodes = 0;
    Stack s;
    s.topIndex = -1;

    for(i = 0; i < MAX_OUTPUTS; i++) {
        unsigned int nodeIndex = c->output[i];
        push(&s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            if( c->nodes[node].active == 0) {
                for (j = 0; j < MAX_ARITY; j++) {
                    if (c->nodes[node].inputs[j] >= N) {
                        push(&s, c->nodes[node].inputs[j] - N);
                    }
                }
                c->activeNodes[c->numActiveNodes] = node;
                c->nodes[node].active = 1;
                c->numActiveNodes++;
            }
        }
    }
}

void mutateTopologyProbabilistic(__global Chromosome *c, __global unsigned int* functionSet, int *seed, int type) {

    int i, j;
    int local_id = get_local_id(0);
    
    for(i = 0; i < MAX_NODES; i++){
    //if(local_id < MAX_NODES) {

        if(randomProb(seed) <= PROB_MUT) {
            c->nodes[i].function = functionSet[randomFunction(seed)];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }

        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(i, seed);
            }
            if(type == 0 && randomProb(seed) <= PROB_MUT){
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
            }
        }
    //}
    }
   // barrier(CLK_LOCAL_MEM_FENCE);
    //if(local_id == 0)
    activateNodes(c);

}

void mutateTopologyProbabilistic2(__global Chromosome *c, __global unsigned int* functionSet, int *seeds, int type) {

    int i, j, k;
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    
    int size = MAX_NODES * MAX_ARITY;

    //printf("%d\n", size);
    //for(i = 0; i < MAX_NODES; i++){
    //if(local_id < MAX_NODES) {

    for(k = 0; k < ceil( (size)/ (float)LOCAL_SIZE_EVOL ) ; k++){

        int index = k * LOCAL_SIZE_EVOL + local_id;
        
        if( index < size){
            int indexNode = index / MAX_ARITY;
            int indexArray = (index % MAX_ARITY);
            //printf("%d\n", indexArray);

            //for(j = 0; j < c->nodes[k * LOCAL_SIZE_EVOL + local_id].maxInputs; j++) {

            if(randomProb(seeds) <= PROB_MUT) {
                c->nodes[indexNode].inputs[indexArray] = randomInput(indexNode, seeds);
            }
            if(type == 0 && randomProb(seeds) <= PROB_MUT){
                c->nodes[indexNode].inputsWeight[indexArray] = randomConnectionWeight(seeds);
            }

            if(indexArray == 0 ) {
                if(randomProb(seeds) <= PROB_MUT ){
                    c->nodes[indexNode].function = functionSet[randomFunction(seeds)];
                    c->nodes[indexNode].maxInputs = getFunctionInputs(c->nodes[indexNode].function);
                }
            }

            //}
        }
    }

    //}
    //}
    barrier(CLK_GLOBAL_MEM_FENCE);
    //if(local_id == 0)
    //    activateNodes(c);

}

void mutateTopologyProbabilisticActive(__global Chromosome *c, __global unsigned int* functionSet, int *seed, int type) {

    int i, j;
    for(i = 0; i < MAX_NODES; i++){
        if(c->nodes[i].active == 1){
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].function = functionSet[randomFunction(seed)];
                c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
            }
            for(j = 0; j < c->nodes[i].maxInputs; j++) {
                if(randomProb(seed) <= PROB_MUT) {
                    c->nodes[i].inputs[j] = randomInput(i, seed);
                }
                if(type == 0 && randomProb(seed) <= PROB_MUT){
                    c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
                }
            }
        }
    }

    activateNodes(c);
}

void mutateTopologyPoint(__global Chromosome *c, __global unsigned int* functionSet, int *seed) {
    int mutationComplete = -1;
    unsigned int newIndex;
    unsigned int newInputIndex;
    unsigned int newValue;

    int num_inputs = MAX_NODES * MAX_ARITY;
    while (mutationComplete == -1){
        unsigned int nodeIndex = randomInterval(0, MAX_NODES + (num_inputs) + MAX_OUTPUTS, seed); //Select any node or output
        if(nodeIndex < MAX_NODES) { // select function
            newIndex = nodeIndex;
            newValue = functionSet[randomFunction(seed)];
            if(newValue != c->nodes[newIndex].function){
                c->nodes[newIndex].function = newValue;
                c->nodes[newIndex].maxInputs = getFunctionInputs(newValue);
                if(c->nodes[newIndex].active > -1) {
                    mutationComplete = 1;
                }
            }
        } else if (nodeIndex <= MAX_NODES + (num_inputs)) { //select input
            newIndex = (unsigned int) ((nodeIndex - MAX_NODES) / MAX_ARITY);
            newInputIndex= (unsigned int) ((nodeIndex - MAX_NODES) % MAX_ARITY);

            newValue = randomInput(newIndex, seed);

            if(newValue != c->nodes[newIndex].inputs[newInputIndex]){
                c->nodes[newIndex].inputs[newInputIndex] = newValue;
                if(c->nodes[newIndex].active == 1) {
                    mutationComplete = 1;
                }
            }

        } else { // select an output
            newIndex = nodeIndex - (MAX_NODES + (num_inputs)) - 1;
            newValue = randomOutputIndex(seed);

            if(newValue != c->output[newIndex]) {
                c->output[newIndex] = newValue;
                mutationComplete = 1;
            }
        }

    }
    activateNodes(c);
}



float executeFunction(__global Chromosome* c, int node, ExStack* exStack){
    int i;
    float result, sum;
    unsigned int inputs = c->nodes[node].maxInputs;
    switch (c->nodes[node].function){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popEx(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popEx(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popEx(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popEx(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popEx(exStack);
            result = pow((float)popEx(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popEx(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popEx(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popEx(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popEx(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popEx(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinear(__global Chromosome* c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    unsigned int inputs = c->nodes[node].maxInputs;
    switch (c->nodes[node].function){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popEx(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popEx(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popEx(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popEx(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popEx(exStack);
            result = pow((float)popEx(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popEx(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popEx(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popEx(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popEx(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popEx(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1.0f + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinearActive(__global ActiveChromosome* c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    unsigned int inputs = c->nodes[node].maxInputs;
    switch (c->nodes[node].function){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popEx(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popEx(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popEx(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popEx(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popEx(exStack);
            result = pow((float)popEx(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popEx(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popEx(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popEx(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popEx(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popEx(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinearActiveImage(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(0,node+1,group_id,0));

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    switch (pixelInt.x){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popEx(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popEx(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popEx(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popEx(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popEx(exStack);
            result = pow((float)popEx(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popEx(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popEx(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popEx(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popEx(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popEx(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));

                sum += (popEx(exStack) * pixelFloat.x);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popEx(exStack) * pixelFloat.x);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popEx(exStack) * pixelFloat.x);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popEx(exStack) * pixelFloat.x);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popEx(exStack) * pixelFloat.x);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}


void evaluateCircuitParallelLinear(__global Chromosome* c,
                                    __global float* data, 
                                    __global float* out, 
                                    __local float* error) {

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[currentActive].inputs[j] - N;

                        
                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        c->fitness = error[0] / M;
    }
}

void evaluateCircuitParallelLinearActive(__global ActiveChromosome* c,
                                        __global float* data, 
                                        __global float* out, 
                                        __local float* error,
                                        __global float* fitness) {
    
    
    
    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
    //ExStackLinear
            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->nodes[i].originalIndex;
                activeInputs = c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[i].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[i].inputs[j] - N;

                        
                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * c->nodes[i].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActive(c, i, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M;

    }
}

void evaluateCircuitParallelTrainValidationLinear(__global Chromosome* c,
                                                __global float* data, 
                                                __global float* out, 
                                                __local float* error) {
                
    
    
    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[currentActive].inputs[j] - N;

                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M_VALIDATION * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

                for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
            
                
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;
            
            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        c->fitnessValidation = error[0] / M_VALIDATION;
    }
}

void evaluateCircuitParallelTrainValidationLinearActive(__global ActiveChromosome* c,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitnessValidation) {
                
    
    
    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->nodes[i].originalIndex;
                activeInputs = c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[i].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[i].inputs[j] - N;

                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M_VALIDATION * c->nodes[i].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActive(c, i, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

                for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
            
                
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;
            
            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        fitnessValidation[group_id] = error[0] / M_VALIDATION;
    }
}




void evaluateCircuitParallelTestLinear(__global Chromosome* c,
                                    __global float* data, 
                                    __global float* out, 
                                    __local float* error) {
    
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TEST/LOCAL_SIZE_TEST) ; k++){

    #else
        for(k = 0; k < ceil( M_TEST/ (float)LOCAL_SIZE_TEST ) ; k++){
            
            if( k * LOCAL_SIZE_TEST + local_id < M_TEST){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
    //ExStackLinear
            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[currentActive].inputs[j] - N;

                        
                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TEST + local_id + ( M_TEST * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TEST + local_id + (M_TEST*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TEST_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TEST_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TEST) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        c->fitness = error[0] / M_TEST;
    }
}

void evaluateCircuitParallelTrainLinear(__global Chromosome* c,
                                    __global float* data, 
                                    __global float* out, 
                                    __local float* error) {
    
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TRAIN/LOCAL_SIZE_TRAIN) ; k++){

    #else
        for(k = 0; k < ceil( M_TRAIN/ (float)LOCAL_SIZE_TRAIN ) ; k++){
            
            if( k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN){
    #endif
        //printf("c");
        //int i, j;
        maxPredicted = -FLT_MAX ;
        predictedClass = 0;
        correctClass = 0;

        for(i = 0; i < MAX_NODES; i++){
            alreadyEvaluated[i] = -FLT_MAX ;
        }

        ExStackLinear exStack;
        exStack.topIndex = -1;


        for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[currentActive].inputs[j] - N;

                        
                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;
        
        #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN) )
    #endif 
           error[local_id] += error[local_id + i];
    }
        
    if(local_id == 0){
        c->fitness = error[0] / M_TRAIN;
    }
}

void evaluateCircuitParallelValidationLinear(__global Chromosome* c,
                                            __global float* data, 
                                            __global float* out, 
                                            __local float* error) {

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
    //ExStackLinear
            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[currentActive].inputs[j] - N;

                        
                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        c->fitnessValidation = error[0] / M_VALIDATION;
    }
}


void evaluateCircuitParallelLinearActiveImage(__read_only image2d_array_t c,
                                        __global float* data, 
                                        __global float* out, 
                                        __local float* error,
                                        __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(c, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;

            //if(group_id == 0 && local_id == 0)
            //    printf("%d \n", pixel.x);

            for(i = 0; i < activenodes; i++){
                int indexCalc = (2*i)+1;
                pixel = read_imageui(c, sampler, (int4)(0, indexCalc, group_id, 0));


                //if(group_id == 0 && local_id == 0)
                //    printf("%d ", pixel.x);


                currentActive = pixel.x;
                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(c, sampler, (int4)(j+1, indexCalc, group_id, 0));


                    //if(group_id == 0 && local_id == 0)
                    //     printf("%d ", pixel.x);


                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.x)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImage(c, indexCalc, &exStack);
                //printf("%f\n",  alreadyEvaluated[currentActive]);
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                pixel = read_imageui(c, sampler, (int4)(i, get_image_height(c)-1, group_id, 0));
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", pixel.x);
                unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M;

    }
}


void evaluateCircuitParallelValidationLinearActiveImage(__read_only image2d_array_t c,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
    //ExStackLinear
            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(c, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            for(i = 0; i < activenodes; i++){
                int indexCalc = (2*i)+1;
                pixel = read_imageui(c, sampler, (int4)(0, indexCalc, group_id, 0));

                currentActive = pixel.x;
                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(c, sampler, (int4)(j+1, indexCalc, group_id, 0));

                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[refIndex]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.x)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImage(c, indexCalc, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                pixel = read_imageui(c, sampler, (int4)(i, get_image_height(c)-1, group_id, 0));

                unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_VALIDATION;

    }
}


__kernel void evolve(__global unsigned int* functionSet,
                     __global int *seeds,
                     __global Chromosome* population,
                     __global Chromosome* best){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    //printf("%d\n", local_id);
    int seed = seeds[global_id];
    //int seed = seeds[group_id];
    //printf("%d\n", local_id);
    //if(local_id == 0)
    //    population[group_id] = *best;

    //barrier(CLK_GLOBAL_MEM_FENCE);
    //mutateTopologyProbabilistic(&newBest[group_id], functionSet, &seed,  0);
    mutateTopologyProbabilistic2(&population[group_id], functionSet, &seed,  0);

    //barrier(CLK_GLOBAL_MEM_FENCE);

    //if(local_id == 0)
    seeds[global_id] = seed;
}



__kernel void evaluateTest(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global Chromosome* individual,
                            __local float* error){


    evaluateCircuitParallelTestLinear(individual, dataset, outputs, error);

}

__kernel void evaluateTrain(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global Chromosome* pop,
                            __local float* error){

    int group_id = get_group_id(0);
    evaluateCircuitParallelTrainLinear(&pop[group_id], dataset, outputs, error);

}

__kernel void evaluateValidation(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global Chromosome* pop,
                            __local float* error){

    int group_id = get_group_id(0);
    evaluateCircuitParallelValidationLinear(&pop[group_id], dataset, outputs, error);

}



__kernel void evaluateTrainValidation(__global float* datasetTrain,
                                        __global float* outputsTrain,
                                        __global float* datasetValid,
                                        __global float* outputsValid,
                                        __global unsigned int* functionSet,
                                        __global Chromosome* pop,
                                        __global Chromosome* best,
                                        __local float* error){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    
    //pop[group_id] = *best;
    //barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitParallelLinear(&pop[group_id], datasetTrain, outputsTrain, error);

    barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitParallelTrainValidationLinear(&pop[group_id], datasetValid, outputsValid, error);

}

__kernel void evaluateTrainValidationActive(__global float* datasetTrain,
                                            __global float* outputsTrain,
                                            __global float* datasetValid,
                                            __global float* outputsValid,
                                            __global unsigned int* functionSet,
                                            __global ActiveChromosome* pop,
                                            __local float* error,
                                            __global float* fitness,
                                            __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    
    //pop[group_id] = *best;
    //barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitParallelLinearActive(&pop[group_id], datasetTrain, outputsTrain, error, fitness);

    barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitParallelTrainValidationLinearActive(&pop[group_id], datasetValid, outputsValid, error, fitnessValidation);

}



__kernel void evaluateTrainImage(__global float* data,
                                __global float* out,
                                __global unsigned int* functionSet,
                                __read_only image2d_array_t pop,
                                __local float* error,
                                __global float* fitness,
                                __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

/*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = read_imageui(pop, sampler, (int4)(1,1,group_id,0));
    if(local_id == 0)
    printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
   */
    //evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);


    evaluateCircuitParallelLinearActiveImage(pop, data, out, error, fitness);

}

__kernel void evaluateValidationImage(__global float* data,
                                    __global float* out,
                                    __global unsigned int* functionSet,
                                    __read_only image2d_array_t pop,
                                    __local float* error,
                                    __global float* fitness,
                                    __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    //evaluateCircuitParallelValidationLinearActiveImage(pop, data, out, error, fitnessValidation);

}







__kernel void testMemory(__global int* test, __local float* error){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
     
    if(local_id == 0)
        printf("%d\n", test[group_id]);
}

