import numpy as np
from guti.parameters import Parameters
from guti.modalities.blur_1d.modality import Blur1D
from copy import deepcopy

param_name = "input_dim"
param_values = [8, 16, 32, 64, 128, 256, 512, 1024]

default_params = Parameters(output_dim=128)

for i, value in enumerate(param_values):
    print(f"-------------------------------- Running {param_name} = {value} ({i+1}/{len(param_values)}) --------------------------------")
    params = deepcopy(default_params)   
    setattr(params, param_name, value)
    modality = Blur1D(params=params)
    modality.run(save_results=True)