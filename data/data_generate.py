import numpy as np
import re

base_dir = [
    'A',
    'G',
    'C',
    'T'
]
# Random generate pesudo-Target
#
# def  generate(file,outputfile):
#     with open(file,encoding='utf-8') as f:
#         with open(outputfile,encoding='utf-8',mode='w') as out:
#             lines = f.readlines()
#             for line in lines:
#                 line = line.strip()
#                 target,microrna = line.split('\t')
#                 microrna = re.sub(r'U','T',microrna)
#                 target_length = len(target)
#                 pesudo_target = []
#                 for i in range(target_length):
#                     index = np.random.randint(1,5)
#                     pesudo_target.append(base_dir[index-1])
#                 pesudo_target = ''.join(pesudo_target)
#                 out.write(target + '\t' + pesudo_target + '\t' + microrna + '\n')
#             out.flush()

# Random generate pesudo-MicroRNA
def  generate(file,outputfile):
    with open(file,encoding='utf-8') as f:
        with open(outputfile,encoding='utf-8',mode='w') as out:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                microrna,target = line.split('\t')
                microrna = re.sub(r'U','T',microrna)
                target_length = len(target)
                mirna_length = len(microrna)
                pesudo_microrna = []
                for i in range(mirna_length):
                    index = np.random.randint(1,5)
                    pesudo_microrna.append(base_dir[index-1])
                pesudo_microrna = ''.join(pesudo_microrna)
                out.write(target + '\t' + pesudo_microrna + '\t' + microrna + '\n')
            out.flush()
generate('T:/microRNA_3/tar_mir.txt','T:\microRNA_3\mti_generate3.txt')
