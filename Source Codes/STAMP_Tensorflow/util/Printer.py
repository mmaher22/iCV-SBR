# coding=utf-8
import time
import pickle as cp
import numpy as np
import sys, os

def TIPrint(samples, config, acc = {}, print_att = False, Time = None):
    base_path = 'output/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    if Time is None:
        suf = time.strftime("%Y%m%d%H%M", time.localtime())
    else:
        suf = Time

    path = base_path + "-" + suf + '.out'
    print_txt(path, samples, config, acc, print_att)
    return suf

def print_txt(path, samples, config, acc = {}, print_att = False):
    outfile = open(path, 'w')
    outfile.write('accuracy:\n')
    for k,v in acc.items():
        outfile.write(str(k) + ' :\t' + str(v) + '\n')

    outfile.write("\nconfig:\n")
    for k,v in config.items():
        outfile.write(str(k) + ' :\t' + str(v) + '\n')

    outfile.write("\nsample:\n")
    for sample in samples:
        outfile.write("id      :\t" + str(sample.id) + '\n')
        outfile.write("session    :\t" + str(sample.session_id) + '\n')
        outfile.write("in_items  :\t" + str(sample.in_idxes) + '\n')
        outfile.write("out_items  :\t" + str(sample.out_idxes) + '\n')
        outfile.write("predict :\t" + str(sample.best_pred) + '\n')
        if print_att:
            for ext_key in sample.ext_matrix:
                matrixs = sample.ext_matrix[ext_key]
                outfile.write("attention :\t" + str(ext_key) + '\n')
                matrix=matrixs[-1]
                for i in range(len(sample.in_idxes)):
                    outfile.write(str(sample.in_idxes[i]) + " :\t")
                    for att in matrix:
                        outfile.write(str(att[i]) + " ")
                    outfile.write("\n")
        outfile.write("\n")
    outfile.close()

def print_binary(path, datas):
    dfile = open(path, 'w')
    cp.dump(datas, dfile)
    dfile.close()
    pass

