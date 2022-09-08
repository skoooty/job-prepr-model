import itertools
from job_prepr_model.ml_logic.params import gridsearch_params

def gridsearch_params_list():
    keys, values = zip(*gridsearch_params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations_dicts
