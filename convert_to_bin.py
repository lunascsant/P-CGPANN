
def read_txt_to_bin(name_file, name_output):
    txt_file = open(name_file, 'r')
    lines = txt_file.readlines()

    infos = [int(i) for i in lines[0].split()]

    inp = infos[0]
    out = infos[1]
    n_lines = infos[2]

    inputs = []
    outputs = []

    for l in range(1, n_lines + 1):
        line = lines[l].split()
        inout = [int(x) for x in line]

        for i in range(inp):
            inputs.append(inout[i])

        for o in range(inp, inp + out):
            outputs.append(inout[o])

    info_bytes = bytearray(infos)
    in_bytes = bytearray(inputs)
    out_bytes = bytearray(outputs)

    output_file = open(name_output, 'wb')
    output_file.write(info_bytes)
    output_file.write(in_bytes)
    output_file.write(out_bytes)
    output_file.close()


def read_bin_to_txt(name_file):
    bin_file = open(name_file, 'rb')
    data_bin = list(bin_file.read(303))

    print(data_bin)
    bin_file.close()    



## MUDE O NOME DO ARQUIVO

read_txt_to_bin('AMH_gsd.txt', 'AMH_gsd.bin')

read_bin_to_txt('AMH_gsd.bin')

