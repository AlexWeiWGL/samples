import numpy as np

def gen_golden_data_simple():
    input_x_ = np.arange(11*15*4).astype(np.float16)*(-1)
    zero_tensor = np.zeros([16-11]).astype(np.float16)
    input_x = np.concatenate((input_x_, zero_tensor),axis=None)   
    golden = np.absolute(input_x).astype(np.float16) 
    input_x.tofile("./input/input_x.bin")      
    golden.tofile("./output/golden.bin")              

if __name__ == '__main__':
    gen_golden_data_simple()