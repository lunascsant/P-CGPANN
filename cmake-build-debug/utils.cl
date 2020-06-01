
typedef struct
{
    unsigned int nodes[MAX_NODES_SIZE];
    unsigned int active[MAX_NODES];
    unsigned int output[MAX_OUTPUT];
    unsigned int numActiveNodes;
    float fitness;

} Circuit;

typedef struct {
    int topIndex;
    unsigned int info[MAX_NODES_SIZE];
} Stack;

void push(Stack* s, unsigned int info){
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES_SIZE){
        s->info[s->topIndex] = info;
    }
}


unsigned int pop(Stack* s){
    if(s->topIndex >= 0){
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    } else {
        return -99;
    }
}

int rand2(int *seed){
    int s  = *seed;
    s = ((unsigned int)(s * 16807) % 2147483647);//(int)(pown(2.0, 31)-1));
    *seed = s;

    return s;
}

unsigned int randomNodeType(int *seed) {
    return rand2(seed) % (NUM_OPBIN + NUM_OPUN + N);
}

unsigned int randomInputVariable(int *seed) {
    return (rand2(seed) % (N)) + NUM_OPBIN + NUM_OPUN;
}

//index: absolute node index (/3)
unsigned int randomInput(unsigned int index, int *seed) {
    return (rand2(seed) % (N + index));
}

unsigned int randomFunction(int *seed) {
    return (rand2(seed) % (NUM_OPBIN + NUM_OPUN));
}

int randomInterval(int inf_bound, int sup_bound, int *seed) {
    return rand2(seed) % (sup_bound - inf_bound + 1) + inf_bound;
}

unsigned int randomOutputIndex(int* seed){
    return (rand2(seed) % MAX_NODES) * 3;
}

/*
*Funcao para retornar um valor Int que representa um Float
*@param floatVal  : valor a ser transformado
*
*@return unsigned int: o inteiro desejado deslocado "NUM_BITS" bits para a direita
*********************************************************/
unsigned int floatToInt(float floatVal){
    return (*(unsigned int*)(&floatVal) >> NUM_BITS);
}


/*
*Funcao para retornar um valor Float que estava salvo em um Int
*@param intVal  : valor a ser transformado
*
*@return float: o float desejado
*********************************************************/
float intToFloat(unsigned int intVal){
    intVal = intVal << NUM_BITS;
    return *(float*)(&intVal);
}


/*
*Funcao para armazenar duas informações em um unico Int
*@param tipo  : primeira informação que deve ter no máxima um tamanho "NUM_BITS" em bits
*       valor : segunda informação (int ou float já transformado em int e deslocado "NUM_BITS" vezes
*
*@return unsigned int: a informação codificada
*********************************************************/
unsigned int packInfo(int tipo, int valor){
    unsigned int informacao = 0;
    informacao = (tipo << (32-NUM_BITS)) | valor;
    return informacao;
}


/*
*Funcao para armazenar duas informações em um unico Int
*@param tipo  : primeira informação que deve ter no máxima um tamanho "NUM_BITS" em bits
*       valor : segunda informação (int)
*
*@return unsigned int: a informação codificada
*********************************************************/
unsigned int packInt(unsigned int tipo, unsigned int valor){
    unsigned int informacao = 0;
    informacao = (tipo << (32-NUM_BITS)) | valor;
    return informacao;
}

/*
*Funcao para armazenar duas informações em um unico Int
*@param tipo  : primeira informação que deve ter no máxima um tamanho "NUM_BITS" em bits
*       valor : segunda informação (float)
*
*@return unsigned int: a informação codificada
*********************************************************/
unsigned int packFloat(int tipo, float valor){
    unsigned int informacao = 0;
    unsigned int valorInt = floatToInt(valor);
    //if(i == 1) printf("tipo ini = %d\n", tipo);

    informacao = (tipo << (32-NUM_BITS)) | valorInt;
    //if(i == 1) printf("tipo ini = %d\n", unpackTipo(informacao));
    return informacao;
}

/*
*Funcao para retornar o 'tipo' de uma informação codificada
*@param info  : informação codificada
*
*@return int: o tipo
*********************************************************/
unsigned int unpackTipo(unsigned int info){
    unsigned int tipo = (info>>(32-NUM_BITS));
    return tipo;
}

/*
*Funcao para retornar um Int de uma informação codificada
*@param info  : informação codificada
*
*@return int: a informação
*********************************************************/
unsigned int unpackInt(unsigned int info){
    int valor = (info << NUM_BITS) >> NUM_BITS;
    return valor;
}

/*
*Funcao para retornar um Float de uma informação codificada
*@param info  : informação codificada
*
*@return float: a informação
*********************************************************/
float unpackFloat(unsigned int info){
    //info = info;
    float valorF = intToFloat(info);
    return valorF;
}


