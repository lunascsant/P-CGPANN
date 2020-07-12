void evaluateCircuitParallel(__global Chromosome* circ,
                            __constant double* data, 
                            __constant double* out, 
                            __local double* error) {
    
    
    __private Chromosome c = (*circ);      
    int i, k, j = 0;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;
    double num, div;

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (double)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
        //printf("c");
        //int i, j;
        double maxPredicted = -DBL_MAX ;
        int predictedClass = 0;
        int correctClass = 0;

        double executionOut[MAX_OUTPUTS];
        double alreadyEvaluated[MAX_NODES];
        int inputsEvaluatedAux[MAX_NODES];

        for(i = 0; i < MAX_NODES; i++){
            alreadyEvaluated[i] = -DBL_MAX ;
            inputsEvaluatedAux[i] = 0;
        }

        Stack s;
        s.topIndex = -1;

        ExStack exStack;
        exStack.topIndex = -1;


        for( i = 0; i < MAX_OUTPUTS; i++) {
            unsigned int nodeIndex = c.output[i];
            push(&s, nodeIndex);

            while(s.topIndex != -1) {
                unsigned int node = pop(&s);
                for (j = inputsEvaluatedAux[node]; j < c.nodes[node].maxInputs; j++) {
                    if (c.nodes[node].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        unsigned int refIndex = c.nodes[node].inputs[j] - N;

                        if(alreadyEvaluated[refIndex] > -DBL_MAX) {
                            inputsEvaluatedAux[node]++;
                            pushEx(&exStack, alreadyEvaluated[refIndex]);
                        } else {
                            push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                            push(&s, refIndex); //avalia o próximo
                            break;
                        }
                    } else {
                        inputsEvaluatedAux[node]++;
                        pushEx(&exStack, data[k * LOCAL_SIZE + local_id + ( M * c.nodes[node].inputs[j])]);
                    }
                }
                if(inputsEvaluatedAux[node] == c.nodes[node].maxInputs){

                  
                    if(!(alreadyEvaluated[node] > -DBL_MAX)) {
                        alreadyEvaluated[node] = executeFunction2(&c, node, &exStack);
                    }
                    //alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                    //pushEx(&exStack, alreadyEvaluated[node]);
                    //alreadyEvaluated[node] = 1;
                }

            }
            executionOut[i] = alreadyEvaluated[nodeIndex];
            /*
            //popEx(&exStack);
            //printf("%f\n", maxPredicted);
            if(executionOut[i] > maxPredicted) {
                maxPredicted = executionOut[i];
                predictedClass = i;
            }

            if(out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0) {
                correctClass = i;
            }
            */
            if(executionOut[i] == out[k*LOCAL_SIZE + local_id + (M*i)]){
                erro+=1.0;
            }
        }
/*
        if(predictedClass == correctClass){
            erro += 1.0;
        }
        */
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

        c.fitness = error[0] / (MAX_OUTPUTS*M);
        circ->fitness = error[0] / (MAX_OUT*M);

    }
}