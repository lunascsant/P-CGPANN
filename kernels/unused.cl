/*
void activateNodesParallel(__global Chromosome* c){

    int local_id = get_local_id(0);

    int i, j;
    int alreadyEvaluated[MAX_NODES];
    if(local_id < MAX_NODES){
        alreadyEvaluated[local_id] = -1;
        c->activeNodes[local_id] = MAX_NODES + 1;
        c->nodes[local_id].active = 0;
    }
    for(i = 0; i < MAX_OUTPUTS; i++){
        c->nodes[c->output[i]].active = 1;
        
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    
    c->numActiveNodes = 0;
    Stack s;
    s.topIndex = -1;
    int numThreads = MAX_OUTPUTS * MAX_ARITY;

    //for(i = 0; i < MAX_OUTPUTS; i++) {
        unsigned int nodeIndex = c->output[local_id / MAX_ARITY];

        c->nodes[nodeIndex].active = 1;
        if (c->nodes[nodeIndex].inputs[local_id % MAX_ARITY] >= N) {
            c->nodes[(local_id % MAX_ARITY) - N].active = 1;
            alreadyEvaluated[(local_id % MAX_ARITY) - N] = local_id;
        }


        //push(&s, nodeIndex);

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
    //}
}
*/

void evaluateCircuitParallel2(__global Chromosome* c,
                                __global float* data,
                                __local float* error) {
    
    

    int i, k, j = 0;
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
        //printf("c");
        //int i, j;
        float maxPredicted = -FLT_MAX ;
        int predictedClass = 0;
        int correctClass = 0;

        float executionOut[MAX_OUTPUTS];
        float alreadyEvaluated[MAX_NODES];
        int inputsEvaluatedAux[MAX_NODES];

        for(i = 0; i < MAX_NODES; i++){
            alreadyEvaluated[i] = -FLT_MAX ;
            inputsEvaluatedAux[i] = 0;
        }

        Stack s;
        s.topIndex = -1;

        ExStack exStack;
        exStack.topIndex = -1;


        for( i = 0; i < MAX_OUTPUTS; i++) {
            unsigned int nodeIndex = c->output[i];
            push(&s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                    if (c->nodes[node].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack, alreadyEvaluated[refIndex]);
                        } else {
                            push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s, refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack, data[k * LOCAL_SIZE + local_id + ( M * c->nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                  
                    if(!(alreadyEvaluated[node] > -FLT_MAX)) {
                        alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    }
                    //alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    //pushEx(&exStack, alreadyEvaluated[node]);
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];//popEx(&exStack);
            //printf("%f\n", maxPredicted);
            if(executionOut[i] > maxPredicted) {
                maxPredicted = executionOut[i];
                predictedClass = i;
            }
            //printf("%f\n", data[k * LOCAL_SIZE + local_id + (M*(N+i))]);
            if(data[k * LOCAL_SIZE + local_id + (M*(N+i))] == 1.0) {
                
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

__kernel void evaluate2(__global float* datasetOut,
                       __global unsigned int* functionSet,
                       __global Chromosome* pop,
                       __local float* error){

    int group_id = get_group_id(0);

    evaluateCircuitParallel2(&pop[group_id], datasetOut, error);

}

void evaluateCircuitParallel(__global Chromosome* c,
                            __global float* data, 
                            __global float* out, 
                            __local float* error) {
    
    
    
    int i, k, j = 0;
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
        //printf("c");
        //int i, j;
        float maxPredicted = -FLT_MAX ;
        int predictedClass = 0;
        int correctClass = 0;

        float executionOut[MAX_OUTPUTS];
        float alreadyEvaluated[MAX_NODES];
        int inputsEvaluatedAux[MAX_NODES];

        for(i = 0; i < MAX_NODES; i++){
            alreadyEvaluated[i] = -FLT_MAX ;
            inputsEvaluatedAux[i] = 0;
        }

        Stack s;
        s.topIndex = -1;

        ExStack exStack;
        exStack.topIndex = -1;


        for( i = 0; i < MAX_OUTPUTS; i++) {
            unsigned int nodeIndex = c->output[i];
            push(&s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                    if (c->nodes[node].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack, alreadyEvaluated[refIndex]);
                        } else {
                            push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s, refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack, data[k * LOCAL_SIZE + local_id + ( M * c->nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                  
                    if(!(alreadyEvaluated[node] > -FLT_MAX)) {
                        alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    }
                    //alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    //pushEx(&exStack, alreadyEvaluated[node]);
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];//popEx(&exStack);
            //printf("%f\n", maxPredicted);
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
void evaluateCircuitParallelTrainValidation(__global Chromosome* c,
                                    __global float* data, 
                                    __global float* out, 
                                    __local float* error) {
    
    

    int i, k, j = 0;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;
    float num, div;

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
            push(&s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                    if (c->nodes[node].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack, alreadyEvaluated[refIndex]);
                        } else {
                            push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s, refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack, data[k * LOCAL_SIZE + local_id + ( M_VALIDATION * c->nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){
                    if(!(alreadyEvaluated[node] > -FLT_MAX)) {
                        alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    }
                    //alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];//popEx(&exStack);
            //printf("%f\n", maxPredicted);
            if(executionOut[i] > maxPredicted) {
                maxPredicted = executionOut[i];
                predictedClass = i;
            }

            if(out[k*LOCAL_SIZE + local_id + (M_VALIDATION*i)] == 1.0) {
                correctClass = i;
            }
        }

        if(predictedClass == correctClass){
            erro += 1.0;
        }
        
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

void evaluateCircuitParallelTest(__global Chromosome* c,
                                __global float* data, 
                                __global float* out, 
                                __local float* error) {
    
    

    int i, k, j = 0;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;
    float num, div;

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
            push(&s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                    if (c->nodes[node].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack, alreadyEvaluated[refIndex]);
                        } else {
                            push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s, refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack, data[k * LOCAL_SIZE_TEST + local_id + ( M_TEST * c->nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                    if(!(alreadyEvaluated[node] > -FLT_MAX)) {
                        alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    }
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];//popEx(&exStack);
            //printf("%f\n", maxPredicted);
            if(executionOut[i] > maxPredicted) {
                maxPredicted = executionOut[i];
                predictedClass = i;
            }

            if(out[k*LOCAL_SIZE_TEST + local_id + (M_TEST*i)] == 1.0) {
                correctClass = i;
            }
            
        }

        if(predictedClass == correctClass){
            erro += 1.0;
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
        
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
        
    if(local_id == 0){
        c->fitness = error[0] / M_TEST;
    }
}

void evaluateCircuitParallelTrain(__global Chromosome* c,
                                    __global float* data, 
                                    __global float* out, 
                                    __local float* error) {
    
    

    int i, k, j = 0;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;
    float num, div;

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
            push(&s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                    if (c->nodes[node].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack, alreadyEvaluated[refIndex]);
                        } else {
                            push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s, refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * c->nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                    if(!(alreadyEvaluated[node] > -FLT_MAX)) {
                        alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    }
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];//popEx(&exStack);
            //printf("%f\n", maxPredicted);
            if(executionOut[i] > maxPredicted) {
                maxPredicted = executionOut[i];
                predictedClass = i;
            }

            if(out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0) {
                correctClass = i;
            }
            
        }

        if(predictedClass == correctClass){
            erro += 1.0;
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
        
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


void evaluateCircuitParallelValidation(__global Chromosome* c,
                                    __global float* data, 
                                    __global float* out, 
                                    __local float* error) {

    int i, k, j = 0;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;
    float num, div;

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
            push(&s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                    if (c->nodes[node].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c->nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -FLT_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack, alreadyEvaluated[refIndex]);
                        } else {
                            push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s, refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * c->nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                    if(!(alreadyEvaluated[node] > -FLT_MAX)) {
                        alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    }
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];//popEx(&exStack);
            //printf("%f\n", maxPredicted);
            if(executionOut[i] > maxPredicted) {
                maxPredicted = executionOut[i];
                predictedClass = i;
            }

            if(out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0) {
                correctClass = i;
            }
            
        }

        if(predictedClass == correctClass){
            erro += 1.0;
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
        
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
        
    if(local_id == 0){
        c->fitnessValidation = error[0] / M_VALIDATION;
    }
}

__kernel void evaluate(__global float* dataset,
                       __global float* outputs,
                       __global unsigned int* functionSet,
                       __global Chromosome* pop,
                       __local float* error){

    int group_id = get_group_id(0);

    evaluateCircuitParallel(&pop[group_id], dataset, outputs, error);

}

__kernel void CGP(__global float* datasetTrain,
                  __global float* outputsTrain,
                  __global float* datasetValid,
                  __global float* outputsValid,
                  __global unsigned int* functionSet,
                  __global int *seeds,
                  __global Chromosome* pop,
                  __global Chromosome* best,
                  __local float* error){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    //printf("%d", group_id);
    int seed = seeds[group_id];
    
    //pop[group_id] = *best;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    //if(local_id == 0){
        mutateTopologyProbabilistic(&pop[group_id], functionSet, &seed,  0);
    //}
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitParallel(&pop[group_id], datasetTrain, outputsTrain, error);

    barrier(CLK_LOCAL_MEM_FENCE);

    evaluateCircuitParallelTrainValidation(&pop[group_id], datasetValid, outputsValid, error);

    

    if(local_id == 0){
        seeds[group_id] = seed;
    }
}
