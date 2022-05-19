# P-CGP
## Steps:
1. Requirements:
Make sure you have OpenCL in your machine and MinGW as a compiler (we recommend the 8.3.0 version).

2. Compile:
PARALLEL
Go to the constants.h file and make sure that the PARALLEL constant is set to 1.
Go to the directory of the project and run in the command line:
g++ *.cpp* -o prog.exe -I"amd/include" -lOpenCl -L"amd/lib/x86_64"
SEQUENTIAL
Go to the constants.h file and make sure that the PARALLEL constant is set to 0.
Go to the directory of the project and run in the command line:
g++ *.cpp* -o prog_seq.exe -I"amd/include" -lOpenCl -L"amd/lib/x86_64"

3. Execute:
Go to the directory of the project and run in the command line:

PARALLEL:
./prog <txt_file_name_genes> <n_execution> <problem_name> <gene_name> <seed>

SEQUENTIAL:
./prog_seq <txt_file_name_genes> <n_execution> <problem_name> <gene_name> <seed>


