/**COMPACT IMAGE_RGBA */

float executeFunctionLinearActiveImageQuarterCompact(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    int local_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(0,node+1,group_id,0));

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    switch (pixelInt.x){

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
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs/4; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) *  pixelFloat.x);
                sum += (popExLinear(exStack) *  pixelFloat.y);
                sum += (popExLinear(exStack) *  pixelFloat.z);
                sum += (popExLinear(exStack) *  pixelFloat.w);

            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif

        default:
            break;
    }
    return result;
}

void evaluateCircuitTrainCompactImage_RGBA(__read_only image2d_array_t indiv,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
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

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            /*if(group_id == 0 && local_id == 0)
                printf("%d %d \n",activenodes, activenodes);
    */

            for(i = 0; i < ceil(activenodes/(float)8); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.z >> 16;
                currentActive1 = pixelAux.z & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.w >> 16;
                currentActive1 = pixelAux.w & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);


                
            }

            for( i = 0; i < MAX_OUTPUTS; i+=4) {
                pixel = read_imageui(indiv, sampler, (int4)(index+1, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*(i+1))] == 1.0)? (i+1) : correctClass;

                if(i+2 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.z;
                if(alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i+2;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*(i+2))] == 1.0)? (i+2) : correctClass;

                if(i+3 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.w;
                if(alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i+3;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*(i+3))] == 1.0)? (i+3) : correctClass;
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
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_TRAIN;

    }
}

void evaluateCircuitValidationCompactImage_RGBA(__read_only image2d_array_t indiv,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
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

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            /*if(group_id == 0 && local_id == 0)
                printf("%d %d \n",activenodes, activenodes);
    */

            for(i = 0; i < ceil(activenodes/(float)8); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.z >> 16;
                currentActive1 = pixelAux.z & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.w >> 16;
                currentActive1 = pixelAux.w & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);


                
            }

            for( i = 0; i < MAX_OUTPUTS; i+=4) {
                pixel = read_imageui(indiv, sampler, (int4)(index+1, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+1))] == 1.0)? (i+1) : correctClass;

                if(i+2 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.z;
                if(alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i+2;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+2))] == 1.0)? (i+2) : correctClass;

                if(i+3 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.w;
                if(alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i+3;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+3))] == 1.0)? (i+3) : correctClass;
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


/**COMPACT IMAGE_RGBA */