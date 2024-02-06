#!/usr/bin/env python3

import parmtSNEcv as p
from ray import tune,train,data
from ray import init as ray_init
import numpy as np

from ray.tune.search import bohb

ray_init()

args = p.parse_args()
adict = p.process_args(args)
adict2 = { **adict }

#XXX

# test data
# clusters = [ 0, 158, 553, 829 ]

# trpcage
clusters = [ 0, 1428, 1625, 1674, 1723 ]



def tune_callback(model,inp,metric):
  cvs = model(inp).cpu().detach().numpy()
  cvs -= np.mean(cvs,axis=0)
  cvs /= np.max(cvs,axis=0) - np.min(cvs,axis=0)

  d = 0.
  for i in range(len(clusters)):
    for j in range(i):
      d += np.linalg.norm(cvs[i]-cvs[j])

  train.report({**metric, 'pairdist' : d })

def tuned(config):

  params = adict2.copy()
  files = [ params['infilename'], params['intopname'] ]

  for f in files:
    ds = data.read_binary_files(f)
    r = ds.take(1)[0]
    with open(f,"wb") as w:
      w.write(r['bytes'])

  for k in params:
    if k in config:
      params[k] = config[k]

# XXX:
  if 'actfun1' not in config: params['actfun1'] = config['actfun']
  if 'actfun2' not in config: params['actfun2'] = config['actfun']
  if 'actfun3' not in config: params['actfun3'] = config['actfun']
  if 'layer1' not in config: params['layer1'] = config['neurons']
  if 'layer2' not in config: params['layer2'] = config['neurons']
  if 'layer3' not in config: params['layer3'] = config['neurons']
#  params.pop('neurons')
#  params.pop('actfun')

  cvs = p.parmtSNEcollectivevariable(**params,report_callback=tune_callback).to('cpu').detach().numpy()
#  d = np.linalg.norm(cvs[14]-cvs[815])
#  return { 'pairdist' : d }
  raise RuntimeError("should never reach here")

space = {
  'lr': tune.loguniform(2e-4,1e-2),
  'layers': tune.choice([1,2,3]),
  'neurons': tune.choice([64,128,256,512]),
  'actfun': tune.choice(['tanh','sigmoid']),
  'shuffle_interval': tune.choice([10,20,50]),
  'batch_size' : tune.choice([256,512,1024,2048]),
  'lagtime' : tune.choice([1,2,3]),
  'perplexity' : tune.choice([15,30,60]),
}

hbbohb = tune.schedulers.HyperBandForBOHB(
  max_t=adict.epochs,
  metric='pairdist',
  mode='max'
)

asha = tune.schedulers.ASHAScheduler(
  max_t=adict.epochs,
  metric='pairdist',
  mode='max',
  grace_period=min(100,adict.epochs),
  reduction_factor=2
)

tunebohb = tune.search.bohb.TuneBOHB(
  metric='pairdist',mode='max'
) 

tuner = tune.Tuner(tune.with_resources(
    tuned,
    resources={'cpu':1, 'gpu':.1 }
  ),
  param_space=space,
  tune_config=tune.TuneConfig(
    num_samples=2000,
    scheduler=asha,
#    scheduler=hbbohb, search_alg=tunebohb
  ),
  run_config=train.RunConfig(
#    storage_path='/work/raytune',
    stop=tune.stopper.TrialPlateauStopper(
      'loss',grace_period=20,
      std=1e-3,num_results=6,
#      metric_threshold = 0.6, mode = 'min'
    )
  ),
)
results = tuner.fit().get_dataframe()
# print(results.get_best_result(metric='pairdist',mode='max').config)

with open('result.json','w') as j:
	results.to_json(j)
