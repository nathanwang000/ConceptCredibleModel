'''
generate a concept file named "concepts_{n_concepts}.txt":
each line of the file is a CUB concept name
it takes an argument --n_concepts 
'''
import os
import sys
import tqdm
import numpy as np

FilePath = os.path.dirname(os.path.abspath(__file__))
RootPath = os.path.dirname(FilePath)
if RootPath not in sys.path: # parent directory
    sys.path = [RootPath] + sys.path
import argparse
from sklearn_extra.cluster import KMedoids

###########
from lib.utils import get_attribute_name, get_class_attributes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputs_dir", default=f"outputs",
                        help="where to save all the outputs")
    parser.add_argument("-c", "--n_concepts", default=[108], type=int, nargs='+',
                        help="list of number of concepts to generate")
        
    args = parser.parse_args()
    print(args)
    return args

def get_medoids_concepts(n_subset, concepts, nruns=1):
    '''
    n_subset: how many concept to sample
    concepts: (n, n_concepts)
    nruns: how many times to run kmedoids

    returns: (n, n_subset)
    '''
    if n_subset >= concepts.shape[1]:
        return concepts
    
    lowest_error_subset, lowest_error = None, np.inf
    for i in tqdm.trange(nruns):
        kmedoids = KMedoids(n_clusters=n_subset,
                            metric='correlation', init='k-medoids++').fit(concepts.T)
        subset = concepts.iloc[:, kmedoids.medoid_indices_]
        abs_sum = np.array(np.abs(subset.corr())).sum()
        if abs_sum < lowest_error: # want correlation to be low
            lowest_error = abs_sum
            lowest_error_subset = subset

    return lowest_error_subset


if __name__ == '__main__':
    flags = get_args()
    savepath = f"{RootPath}/{flags.outputs_dir}/concepts"
    print(savepath)

    class_attributes = get_class_attributes()
    # CBM paper report 112 concepts, but we have 108 concepts
    maj_concepts = class_attributes.loc[:, ((class_attributes >= 50).sum(0) >= 10)] >= 50

    for n_concept in flags.n_concepts:
        attr_names = list(get_medoids_concepts(n_concept, maj_concepts).columns)
        with open(savepath + f"_{n_concept}.txt", "w") as f:
            for name in attr_names:
                f.write(f"{name}\n")

        
