#!/usr/bin/env python3

import parmtSNEcv as p
from ray import tune
import numpy as np

args = p.parse_args()
adict = p.process_args(args)


def tuned(config):
  params = adict.copy()
  for k in params:
    if k in config:
      params[k] = config[k]

  cvs = p.parmtSNEcollectivevariable(**params)
	d = np.linalg.norm(cvs[14]-cvs[815])
	return { 'pairdist' : d }


space = {
  'layers': tune.choice([1,2,3]),
	'layer1': tune.choice([32,64,128,256]),
	'layer2': tune.sample_from(lambda x: x.config.layer1),
	'layer3': tune.sample_from(lambda x: x.config.layer1),
}

tuner = tune.Tuner(tuned, param_space = space)
results = tuner.fit()
print(results.get_best_result(metric='pairdist','max').config)
