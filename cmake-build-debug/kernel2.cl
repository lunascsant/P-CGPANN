#include "utils.cl"

void pushNodeStack_test(__global Circuit* c, Stack* s, unsigned int index){
    push(s, packInt(INDEX, index));
    push(s, c->nodes[index+1]);
    push(s, c->nodes[index+2]);

}

void pushNodeStack_test2(Circuit* c, Stack* s, unsigned int index){
    push(s, packInt(INDEX, index));
    push(s, c->nodes[index+1]);
    push(s, c->nodes[index+2]);

}

void activateNodes(Circuit* c){

    int i;
    int alreadyEvaluated[MAX_NODES] = {-1};
    for(i = 0; i < MAX_NODES; i++) {
        alreadyEvaluated[i] = -1;
        c->active[i] = 0;
    }
    c->numActiveNodes = 0;

    Stack s;
    s.topIndex = -1;

    unsigned int activeIndexValue;

    for(i = 0; i < MAX_OUTPUT; i++) {
        unsigned int nodeIndex = c->output[i];
        pushNodeStack_test2(c, &s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            unsigned int nodeType = unpackTipo(node);
            unsigned int nodeInfo = unpackInt(node);
            //unsigned int aux_node, aux_nodeType, aux_nodeInfo;

            if(nodeType == INDEX) {
                activeIndexValue = nodeInfo/3; // node being executed now
                if(c->active[activeIndexValue] != 1) {
                    c->active[activeIndexValue] = 1; //activate
                    unsigned int function =  unpackInt(c->nodes[nodeInfo]);
                    switch(function){
                        case OR:
                        case AND:
                        case NAND:
                            c->numActiveNodes += 2;
                            break;
                        case NOT:
                        case NOR:
                            c->numActiveNodes += 1;
                            break;
                        case XOR:
                            c->numActiveNodes += 3;
                            break;
                        case XNOR:
                            c->numActiveNodes += 4;
                            break;
                        default:
                            //std::cout << "erro" << std::endl;
                            break;
                    }
                    //alreadyEvaluated[activeIndexValue] = 1;
                }

            } else if (nodeType == NODE) { //if it's a reference to another node, push it to the stack
                //if(alreadyEvaluated[nodeInfo/3] != 1){
                pushNodeStack_test2(c, &s, nodeInfo);
                //}
            }
        }
    }
    //std::cout<<std::endl;
}

void *mutate(Circuit *c, int *seed, __global unsigned int* functionSet) {
    int mutationComplete = -1;
    int newActivePath = -1;
    while (mutationComplete == -1){
        unsigned int nodeIndex = randomInterval(0, MAX_NODES_SIZE + MAX_OUTPUT, seed); //Select any node or output

        if(nodeIndex < MAX_NODES_SIZE){ // select a node
            unsigned int startIndex = (nodeIndex - (nodeIndex % 3) )/ 3;


            unsigned int nodeType = unpackTipo(c->nodes[nodeIndex]);
            if (nodeType == FUN || nodeType == FBIN) {
                unsigned int newFunction = functionSet[randomFunction(seed)];

                while(newFunction == c->nodes[nodeIndex]) newFunction = functionSet[randomFunction(seed)];

                c->nodes[nodeIndex] = newFunction;

                if(unpackTipo(c->nodes[nodeIndex]) == FUN) {
                    c->nodes[nodeIndex + 2] = c->nodes[nodeIndex + 1];
                }

            } else if (nodeType == NODE || nodeType == VAR) {


                unsigned int newValue = randomInput(startIndex, seed);
                while (newValue == unpackInt(c->nodes[nodeIndex])) newValue = randomInput(startIndex, seed);

                if (newValue < startIndex) {
                    //if(c->active[newValue] == 0) newActivePath = 1;
                    newValue = packInt(NODE, newValue * 3);
                    c->nodes[nodeIndex] = newValue;

                } else {
                    newValue = packInt(VAR, newValue - startIndex);
                    c->nodes[nodeIndex] = newValue;
                }
                unsigned int nodeFunctionType = unpackTipo(c->nodes[nodeIndex - (nodeIndex % 3)]);
                if(nodeFunctionType == FUN){
                    c->nodes[nodeIndex - (nodeIndex % 3) + 1] = newValue;
                    c->nodes[nodeIndex - (nodeIndex % 3) + 2] = newValue;
                }
            }
            if(c->active[startIndex] == 1){
                mutationComplete = 1;
            }
        } else { // select an output
            unsigned int outputIndex = nodeIndex - MAX_NODES_SIZE;
            unsigned int newOutput = randomOutputIndex(seed);
            while (newOutput == c->output[outputIndex]){
                newOutput = randomOutputIndex(seed);
            }
            c->output[outputIndex] = newOutput;
            //if(c->active[outputIndex/3] == 0) newActivePath = 1;

            mutationComplete = 1;
        }
    }

    activateNodes(c);
}

void evaluateCircuitParallel(__global Circuit* c,
                            __global unsigned int* data, 
                            __global unsigned int* out, 
                            __local float* error) {
    
    //int i, k;
    c->fitness = 0.0;

    int i, k = 0;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;
    float num, div;

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
    
        int i;
        unsigned int executionOut[MAX_OUTPUT];
        int alreadyEvaluated[MAX_NODES];
        for(i = 0; i < MAX_NODES; i++) alreadyEvaluated[i] = -1;

        Stack s, exStack;
        s.topIndex = -1;
        exStack.topIndex = -1;

        unsigned int activeIndexValue = -1;

        for( i = 0; i < MAX_OUTPUT; i++) {
            unsigned int nodeIndex = c->output[i];
            pushNodeStack_test(c, &s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                unsigned int nodeType = unpackTipo(node);
                unsigned int nodeInfo = unpackInt(node);

                if(nodeType == INDEX) {
                    node = c->nodes[nodeInfo];
                    activeIndexValue = nodeInfo/3; // node being executed now

                    nodeType = unpackTipo(node);
                    nodeInfo = unpackInt(node);
                }

                if(nodeType == FUN || nodeType == FBIN){
                    unsigned int result;
                    switch (nodeInfo){
                        case AND:
                            result = pop(&exStack) & pop(&exStack);
                            push(&exStack, result);
                            break;
                        case OR:
                            result = pop(&exStack) | pop(&exStack);
                            push(&exStack, result);
                            break;
                        case XOR:
                            result = pop(&exStack) ^ pop(&exStack);
                            push(&exStack, result);
                            break;
                        case NAND:
                            result = ~(pop(&exStack) & pop(&exStack)) & 1;
                            push(&exStack, result);
                            break;
                        case NOR:
                            result = ~(pop(&exStack) | pop(&exStack)) & 1;
                            push(&exStack, result);
                            break;
                        case XNOR:
                            result = ~(pop(&exStack) ^ pop(&exStack)) & 1;
                            push(&exStack, result);
                            break;
                        case NOT:
                            pop(&exStack);
                            result = ~(pop(&exStack)) & 1;
                            push(&exStack, result);
                            break;
                        default:
                            break;
                    }
                    alreadyEvaluated[activeIndexValue] = result;

                } else if (nodeType == NODE) { //if it's a reference to another node, push it to the stack
                    if(alreadyEvaluated[nodeInfo/3] != -1){
                        push(&exStack, alreadyEvaluated[nodeInfo/3]);
                    } else {
                        pushNodeStack_test(c, &s, nodeInfo);
                    }
                } else { //VAR
                    push(&exStack, data[k*LOCAL_SIZE+local_id+(M*nodeInfo)]);
                }
            }
            executionOut[i] = pop(&exStack);
         //if(local_id == 0)
         //printf("i = %d \n",i);

            //if(group_id == 0 && local_id == 0) printf("i = %d ,local_id = %d, index = %d \n", i, local_id, k*LOCAL_SIZE+local_id+ (M*i));


            if(executionOut[i] == out[k*LOCAL_SIZE+local_id+ (M*i)]){

                erro += 1.0;
            }
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
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
        
    if(local_id == 0){
        c->fitness = error[0]; 
    }
}


__kernel void evolve(Circuit best,
                     __global Circuit* newBest,
                     __global unsigned int* functionSet,
                     __global int *seeds){

    int group_id = get_group_id(0);

    int seed = seeds[group_id];

    mutate(&best, &seed, functionSet);

    //remaining cores evaluate
    //evaluateCircuit(&best, dataset, outputs);
    //evaluateCircuitParallel(&mutated_best, dataset, outputs, error);
    //barrier(CLK_LOCAL_MEM_FENCE);

    newBest[group_id] = best;
    /*
    if(mutated_best.fitness > best.fitness){
        newBest[group_id] = mutated_best;
    } else if (mutated_best.fitness == best.fitness) {
        if(mutated_best.numActiveNodes <= best.numActiveNodes){
            newBest[group_id] = mutated_best;
        }
    }
     */

    seeds[group_id] = seed;
}

__kernel void evaluate(__global Circuit* pop,
                       __global unsigned int* dataset,
                       __global unsigned int* outputs,
                       __global unsigned int* functionSet,
                       __local float* error){
    //printf("a");
    int group_id = get_group_id(0);
    //Circuit c = pop[group_id];
    //remaining cores evaluate
    //evaluateCircuit(&best, dataset, outputs);
    evaluateCircuitParallel(&pop[group_id], dataset, outputs, error);
    //barrier(CLK_LOCAL_MEM_FENCE);

    //pop[group_id] = c;
    /*
    if(mutated_best.fitness > best.fitness){
        newBest[group_id] = mutated_best;
    } else if (mutated_best.fitness == best.fitness) {
        if(mutated_best.numActiveNodes <= best.numActiveNodes){
            newBest[group_id] = mutated_best;
        }
    }
     */

}
