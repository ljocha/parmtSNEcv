#!/usr/bin/env python3

import parmtSNEcv as p
from ray import tune
import numpy as np

args = p.parse_args()
adict = p.process_args(args)
adict2 = { **adict }


def tuned(config):
  params = adict2.copy()
  for k in params:
    if k in config:
      params[k] = config[k]

# XXX:
  if 'actfun2' not in config: params['actfun2'] = params['actfun1']
  if 'actfun3' not in config: params['actfun3'] = params['actfun1']
  if 'layer2' not in config: params['layer2'] = params['layer1']
  if 'layer3' not in config: params['layer3'] = params['layer1']

  cvs = p.parmtSNEcollectivevariable(**params).to('cpu').detach().numpy()
  d = np.linalg.norm(cvs[14]-cvs[815])
  return { 'pairdist' : d }

def test(config):
  return { 'score', 1 }


space = {
  'layers': tune.choice([1,2,3]),
  'layer1': tune.choice([32,64,128,256]),
  'actfun1': tune.choice(['tanh','sigmoid']),
}


tuner = tune.Tuner(tuned, param_space=space)
#tuner = tune.Tuner(test, param_space=space)
results = tuner.fit()
print(results.get_best_result(metric='pairdist',mode='max').config)
