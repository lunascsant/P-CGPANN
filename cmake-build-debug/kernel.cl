#include "utils.cl"

void pushNodeStack_test(Circuit* c, Stack* s, unsigned int index){
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
        pushNodeStack_test(c, &s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            unsigned int nodeType = unpackTipo(node);
            unsigned int nodeInfo = unpackInt(node);
            //unsigned int aux_node, aux_nodeType, aux_nodeInfo;

            if(nodeType == INDEX) {
                activeIndexValue = nodeInfo/3; // node being executed now
                if(c->active[activeIndexValue] != 1) {
                    c->active[activeIndexValue] = 1; //activate
                    c->numActiveNodes++;
                    //alreadyEvaluated[activeIndexValue] = 1;
                }

            } else if (nodeType == NODE) { //if it's a reference to another node, push it to the stack
                //if(alreadyEvaluated[nodeInfo/3] != 1){
                pushNodeStack_test(c, &s, nodeInfo);
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

void evaluateCircuit(Circuit* c, __global unsigned int* data, __global unsigned int* out) {
    
    int i, k;
    c->fitness = 0.0;
    for(k = 0; k < M; k++){
        //runCircuit_Test(c, p, data[i], out[i]);
        int i;
        unsigned int executionOut[MAX_OUTPUT];
        int alreadyEvaluated[MAX_NODES] = {-1};
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
                    push(&exStack, data[k+M*nodeInfo]);
                }
            }
            executionOut[i] = pop(&exStack);

            if(executionOut[i] == out[k+ M*i]){
                c->fitness += 1.0;
            }
        }
    }
}

void runCircuit_Test(Circuit* c, unsigned int* data, unsigned int* out){

    int i;
    unsigned int executionOut[MAX_OUTPUT];
    int alreadyEvaluated[MAX_NODES] = {-1};
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
                push(&exStack, data[nodeInfo]);
            }
        }
        executionOut[i] = pop(&exStack);

        if(executionOut[i] == out[i]){
            c->fitness += 1.0;
        }
    }
}

__kernel void evolve(Circuit best,
                     __global Circuit* newBest,
                     __global unsigned int* dataset,
                     __global unsigned int* outputs,
                     __global unsigned int* functionSet,
                     __global int *seeds){

    int group_id = get_group_id(0);

    int seed = seeds[group_id];

    mutate(&best, &seed, functionSet);
    
    evaluateCircuit(&best, dataset, outputs);

    newBest[group_id] = best;

    seeds[group_id] = seed;
}