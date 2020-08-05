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
    unsigned int numActiveNodes;
    float fitness;
    float fitnessValidation;
} Chromosome;

typedef struct {
    int topIndex;
    unsigned int info[MAX_NODES * MAX_ARITY];
} Stack;

typedef struct {
    int topIndex;
    float info[MAX_NODES * MAX_ARITY];
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

void pushEx(ExStack* s  float info) {
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

float randomConnectionWeight(int *seed) {
    return (((float) rand2(seed) / (float) (2147483647) ) * 2 * WEIGTH_RANGE) - WEIGTH_RANGE;
}

int randomInterval(int inf_bound  int sup_bound  int *seed) {
    return rand2(seed) % (sup_bound - inf_bound + 1) + inf_bound;
}

float randomProb(int* seed){
    return (float)rand2(seed) / 2147483647;//pown(2.0  31);
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
    int i  j;
    int alreadyEvaluated[MAX_NODES] = {-1};
    for(i = 0; i < MAX_NODES; i++) {
        alreadyEvaluated[i] = -1;
        c->nodes[i].active = 0;
    }

    c->numActiveNodes = 0;
    Stack s;
    s.topIndex = -1;

    for(i = 0; i < MAX_OUTPUTS; i++) {
        unsigned int nodeIndex = c->output[i];
        push(&s  nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            if( c->nodes[node].active == 0) {
                for (j = 0; j < MAX_ARITY; j++) {
                    if (c->nodes[node].inputs[j] >= N) {
                        push(&s  c->nodes[node].inputs[j] - N);
                    }
                }
                c->nodes[node].active = 1;
                c->numActiveNodes++;
            }

        }
    }
}

void mutateTopologyProbabilistic(__global Chromosome *c  __global unsigned int* functionSet  int *seed  int type) {

    int i  j;
    for(i = 0; i < MAX_NODES; i++){

        if(randomProb(seed) <= PROB_MUT) {
            c->nodes[i].function = functionSet[randomFunction(seed)];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }
        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(i  seed);
            }
            if(type == 0 && randomProb(seed) <= PROB_MUT){
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
            }
        }
    }

    activateNodes(c);
}

void mutateTopologyProbabilisticActive(__global Chromosome *c  __global unsigned int* functionSet  int *seed  int type) {

    int i  j;
    for(i = 0; i < MAX_NODES; i++){
        if(c->nodes[i].active == 1){
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].function = functionSet[randomFunction(seed)];
                c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
            }
            for(j = 0; j < c->nodes[i].maxInputs; j++) {
                if(randomProb(seed) <= PROB_MUT) {
                    c->nodes[i].inputs[j] = randomInput(i  seed);
                }
                if(type == 0 && randomProb(seed) <= PROB_MUT){
                    c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
                }
            }
        }
    }

    activateNodes(c);
}

void mutateTopologyPoint(__global Chromosome *c  __global unsigned int* functionSet  int *seed) {
    int mutationComplete = -1;
    unsigned int newIndex;
    unsigned int newInputIndex;
    unsigned int newValue;

    int num_inputs = MAX_NODES * MAX_ARITY;
    while (mutationComplete == -1){
        unsigned int nodeIndex = randomInterval(0  MAX_NODES + (num_inputs) + MAX_OUTPUTS  seed); //Select any node or output
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

            newValue = randomInput(newIndex  seed);

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

float executeFunction(__global Chromosome* c  int node  ExStack* exStack){
    int i;
    float result  sum;
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
            result = pow((float)popEx(exStack)  (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popEx(exStack)  (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popEx(exStack);
            result = pow((float)popEx(exStack)  (float)result);
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
            result = 1 / (1 + exp((-1) * sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow((float) (sum - 0)  (float) 2)) / (2 * pow((float)1  (float)2)));
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

void evaluateCircuitParallel(__global Chromosome* c 
                            __constant float* data
                            __constant float* out
                            __local float* error) {
    
    //c->fitness = 0.0;

    int i  k  j = 0;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;
    float num  div;

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
        //int i  j;
        float maxPredicted = -FLT_MAX;
        int predictedClass = 0;
        int correctClass = 0;

        float executionOut[MAX_OUTPUTS];
        float alreadyEvaluated[MAX_NODES];
        int inputsEvaluatedAux[MAX_NODES];

        for(i = 0; i < MAX_NODES; i++){
            alreadyEvaluated[i] = -FLT_MAX;
            inputsEvaluatedAux[i] = 0;
        }

        Stack s;
        s.topIndex = -1;

        ExStack exStack;
        exStack.topIndex = -1;


        for( i = 0; i < MAX_OUTPUTS; i++) {
            unsigned int nodeIndex = c->output[i];
            push(&s  nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                    if (c->nodes[node].inputs[j] >= N) { // se é um outro nó  empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack  alreadyEvaluated[refIndex]);
                        } else {
                            push(&s  node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s  refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack  data[k * LOCAL_SIZE + local_id + ( M * c->nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                    alreadyEvaluated[node] = executeFunction(c  node  &exStack);
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];//popEx(&exStack);
            //printf("%f\n"  maxPredicted);
            if(executionOut[i] > maxPredicted) {
                maxPredicted = executionOut[i];
                predictedClass = i;
            }

            if(out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0) {
                correctClass = i;
            }
        }

        if(predictedClass == correctClass){
            erro += 1.0;
        }
        
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
        /* LOCAL_SIZE is not power of 2  so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
        
    if(local_id == 0){
        #ifdef IS_VALIDATION
            c->fitnessValidation = error[0] / M;
        #else
            c->fitness = error[0] / M;
        #endif
    }
}


__kernel void evolve(__global Chromosome* best 
                     __global Chromosome* newBest 
                     __global unsigned int* functionSet 
                     __global int *seeds){

    int group_id = get_group_id(0);
    
    int seed = seeds[group_id];

    newBest[group_id] = *best;

    barrier(CLK_LOCAL_MEM_FENCE);

    mutateTopologyProbabilistic(&newBest[group_id]  functionSet  &seed   0);

    barrier(CLK_LOCAL_MEM_FENCE);

    
    seeds[group_id] = seed;
}

__kernel void evaluate(__global Chromosome* pop 
                       __constant float* dataset
                       __constant float* outputs
                       __global unsigned int* functionSet 
                       __local float* error){

    int group_id = get_group_id(0);

    evaluateCircuitParallel(&pop[group_id]  dataset  outputs  error);

}


__kernel void CGP(__global Chromosome* pop 
                  __global Chromosome* best 
                  __constant float* dataset
                  __constant float* outputs
                  __global unsigned int* functionSet 
                  __global int *seeds 
                  __local float* error){

    int group_id = get_group_id(0);
    int seed = seeds[group_id];

    pop[group_id] = *best;
    barrier(CLK_LOCAL_MEM_FENCE);

    mutateTopologyProbabilistic(&pop[group_id]  functionSet  &seed   0);

    barrier(CLK_LOCAL_MEM_FENCE);

    evaluateCircuitParallel(&pop[group_id]  dataset  outputs  error);

    barrier(CLK_LOCAL_MEM_FENCE);

    seeds[group_id] = seed;
}