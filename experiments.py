'''
a setting is a list of argument like the following
[('--lr', 0.001), ('--wd', 0), '--pmt', ('--runname', runname)]
'''
import numpy as np
import os
import argparse

# custom import
from lib.tune import run

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputs_dir", default="outputs",
                        help="where outputs are saved")
    parser.add_argument("--nc", default=1, type=int,
                        help="number of concurrent jobs, default 1")
    args = parser.parse_args()
    print(args)
    return args

def experiment(flags):
    tasks = [
        [('--hidden_dim', 100)],
        [('--hidden_dim', 5)]        
    ]
    # without gpu just uses cpu
    run("main.py", tasks, gpus=[], n_concurrent_process=flags.nc,
        o_dir=flags.outputs_dir)

if __name__ == '__main__':
    flags = get_args()
    experiment(flags)
