'''
---------------------------------------------
--- Jose Eduardo H. da Silva - 06/23/2022 ---
---------------------------------------------
Usage: python TTConverter.py CircuitName (ex.: python TTConverter.py C17)

CircuitName File must be in the form of sum of products, using .EP format

File Example:

.p 32
.i 5
.o 2
i0*i1*i2*i3
i1*i2*i3*i4+i1*i2


File Description:
.p is the number of combinations (2^number of inputs)
.i is the number of primary inputs
.o is the number of primary outputs

i0*i1*i2*i3 -> means the product between the inputs i0, i1, i2 and i3, where i0 is the most significative bit
i1*i2*i3*i4+i1*i2 -> two minterms: i1*i2*i3*i4 and i1*i2

Output File has the same name of the input file, but with .tt extension

Output File Example:

5 2 32
0 0 0 0 0 0 0 
0 0 0 0 1 0 0 
0 0 0 1 0 0 0 
0 0 0 1 1 0 0 
0 0 1 0 0 0 0 
0 0 1 0 1 0 0 
0 0 1 1 0 0 0 
0 0 1 1 1 0 0 
0 1 0 0 0 0 0 
0 1 0 0 1 0 0 
0 1 0 1 0 0 0 
0 1 0 1 1 0 0 
0 1 1 0 0 0 1 
0 1 1 0 1 0 1 
0 1 1 1 0 0 1 
0 1 1 1 1 0 1 
1 0 0 0 0 0 0 
1 0 0 0 1 0 0 
1 0 0 1 0 0 0 
1 0 0 1 1 0 0 
1 0 1 0 0 0 0 
1 0 1 0 1 0 0 
1 0 1 1 0 0 0 
1 0 1 1 1 0 0 
1 1 0 0 0 0 0 
1 1 0 0 1 0 0 
1 1 0 1 0 0 0 
1 1 0 1 1 0 0 
1 1 1 0 0 0 1 
1 1 1 0 1 0 1 
1 1 1 1 0 1 1 
1 1 1 1 1 1 1

Header: 5 2 32 -> number of inputs, number of outputs, number of combinations
The length of each row is the number of inputs + the number of outputs ( 5 + 2 )
Each row has the combination of inputs (length = number of inputs) followed by the output value (length = number of outputs)
For example, the combination of inputs 0 0 0 0 0, gives 0 at output0 and 0 at output1.
In the same way, 1 1 1 0 1, gives 0 at output0 and 1 at output1.

'''

FILENAME = "tables/cordic"

print("Initializing " + FILENAME)

OUTPUT_EXPRESSIONS = []
GLOBAL_NI = [1]
GLOBAL_NO = [1]
GLOBAL_COMBINATIONS = [1]
INPUTS = []
OUTPUTS = []


def decimalToBinary(n):

    binario = "{0:b}".format(int(n))

    valor = ""
    if len(binario) < GLOBAL_NI[0]:
        iteracoes = GLOBAL_NI[0] - len(binario)
        for i in range(iteracoes):
            valor += "0"
        #print("IT: ", iteracoes)
        valor += str(binario)
    else:
        valor = str(binario)
    
    return valor

def generate_Outputs(currentOutputExpression):
    if len(currentOutputExpression) == 1:
        currentOutputExpression = currentOutputExpression[0]
        neededInputs = []
        for outputIndex in currentOutputExpression:
            neededInputs.append(INPUTS[outputIndex])
        OUTPUTS.append([])
        currentOutputIndex = len(OUTPUTS)-1
        for i in range(len(neededInputs[0])):
            currentProduct = neededInputs[0][i]
            for j in range(len(neededInputs)):
                    currentProduct *= neededInputs[j][i]
            #print(currentProduct)
            OUTPUTS[currentOutputIndex].append(currentProduct)
    else:
        allProducts = []
        for allExpressions in currentOutputExpression:
            neededInputs = []
            for outputIndex in allExpressions:
                neededInputs.append(INPUTS[outputIndex])
            allProducts.append([])
            currentProductIndex = len(allProducts)-1
            for i in range(len(neededInputs[0])):
                currentProduct = neededInputs[0][i]
                for j in range(len(neededInputs)):
                    currentProduct *= neededInputs[j][i]
                allProducts[currentProductIndex].append(currentProduct)
        
        OUTPUTS.append([])
        currentOutputIndex = len(OUTPUTS)-1
        for i in range(len(allProducts[0])):
            currentSum = allProducts[0][i]
            for j in range(len(allProducts)):
                auxSum = (currentSum or allProducts[j][i])
                currentSum = auxSum
            OUTPUTS[currentOutputIndex].append(currentSum)
                

def generateInputBits():
    for entradas in range(GLOBAL_NI[0]):
        INPUTS.append([])
        proporcao = int(pow(2, GLOBAL_NI[0]) / pow(2, entradas + 1))
        for divisoes in range(pow(2, entradas + 1)):
            for bits in range(proporcao):
                if divisoes % 2 == 0:
                    INPUTS[entradas].append(0)
                else:
                    INPUTS[entradas].append(1)

    for entradas in range(GLOBAL_NI[0]):
        INPUTS.append([])
        proporcao = int(pow(2, GLOBAL_NI[0]) / pow(2, entradas + 1))
        for divisoes in range(pow(2, entradas + 1)):
            for bits in range(proporcao):
                if divisoes % 2 == 0:
                    INPUTS[entradas+GLOBAL_NI[0]].append(1)
                else:
                    INPUTS[entradas+GLOBAL_NI[0]].append(0)    



fileEP = open(FILENAME+'.ep', 'r')
for line in fileEP:
    if line.find(".i") != -1:
        GLOBAL_NI[0] = int(line.split(" ")[1].strip())
    elif line.find(".o") != -1:
        GLOBAL_NO[0] = int(line.split(" ")[1].strip())
    elif line.find(".p") != -1:
        GLOBAL_COMBINATIONS[0] = int(line.split(" ")[1].strip())
    else:
        splitLine = line.split('+')
        #splitLine = line.split('*')
        if len(splitLine) > 0:
            currentGeneralExpression = []
            for inputs in splitLine:
                currentExpression = []
                splitByPlus = inputs.split('*')
                for currentSplit in splitByPlus:
                    replaceSymbol = currentSplit.strip().replace('i', '')#inputs.strip().replace('i', '')
                    #print(replaceSymbol)
                    if replaceSymbol.find('~') != -1:
                        #print(replaceSymbol)
                        #print(replaceSymbol[1])
                        replaceSymbol = int(replaceSymbol[1]) + GLOBAL_NI[0]
                    currentExpression.append(int(replaceSymbol))
                currentGeneralExpression.append(currentExpression)
            #OUTPUT_EXPRESSIONS.append(currentExpression)
            OUTPUT_EXPRESSIONS.append(currentGeneralExpression)
            

print("Generating Input Bits...")
generateInputBits()
print("Generating Output Expressions...")
for expression in OUTPUT_EXPRESSIONS:
    generate_Outputs(expression)



print("Generating Output File...")
outputFile = open(FILENAME+'.tt', 'w')
header = str(GLOBAL_NI[0]) + " " + str(GLOBAL_NO[0]) + " " + str(GLOBAL_COMBINATIONS[0]) + "\n"
outputFile.write(header)

for i in range(pow(2, GLOBAL_NI[0])):
    currentInput = decimalToBinary(i)
    for output in OUTPUTS:
        currentInput += str(output[i])
    currentLine = ""
    for bits in currentInput:
        currentLine += bits
        currentLine += " "
    currentLine += "\n"
    outputFile.write(currentLine)
outputFile.close()

print(FILENAME + " sucessfully converted.")
