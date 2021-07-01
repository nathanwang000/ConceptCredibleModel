'''
a setting is a list of argument like the following
[('--lr', 0.001), ('--wd', 0), '--pmt', ('--runname', runname)]
'''
import numpy as np
import os
import argparse
import sys

# custom import
FilePath = os.path.dirname(os.path.abspath(__file__))
RootPath = os.path.dirname(FilePath)
if RootPath not in sys.path: # parent directory
    sys.path = [RootPath] + sys.path
from lib.tune import run

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", default=1, type=int,
                        help="number of concurrent jobs, default 1")
    args = parser.parse_args()
    print(args)
    return args

def experiment(flags):
    tasks = []
    for n_concept in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 108]:
        tasks.append([("--concept_path", f"outputs/concepts/concepts_{n_concept}.txt")])

    run("./scripts/ground_truth.py", tasks, gpus=[0, 1, 2, 3, 4, 6, 7], n_concurrent_process=flags.nc,
        track=False)

if __name__ == '__main__':
    flags = get_args()
    experiment(flags)
