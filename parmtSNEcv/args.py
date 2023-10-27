import argparse as arg
import sys

def parse_args():

  parser = arg.ArgumentParser(description='Parametric t-SNE using artificial neural networks for development of collective variables of molecular systems, requires numpy, keras and mdtraj')
  
  parser.add_argument('-i', dest='infile', default='traj.xtc',
  help='Input trajectory in pdb, xtc, trr, dcd, netcdf or mdcrd, WARNING: the trajectory must be 1. centered in the PBC box, 2. fitted to a reference structure and 3. must contain only atoms to be analysed!')
  
  parser.add_argument('-p', dest='intop', default='top.pdb',
  help='Input topology in pdb, WARNING: the structure must be 1. centered in the PBC box and 2. must contain only atoms to be analysed!')
  
  parser.add_argument('-dim', dest='embed_dim', default=2, type=int,
  help='Number of output dimensions (default 2)')
  
  parser.add_argument('-perplex', dest='perplex', default=30., type=float,
  help='Value of t-SNE perplexity (default 30.0)')
  
  parser.add_argument('-boxx', dest='boxx', default=0.0, type=float,
  help='Size of x coordinate of PBC box (from 0 to set value in nm)')
  
  parser.add_argument('-boxy', dest='boxy', default=0.0, type=float,
  help='Size of y coordinate of PBC box (from 0 to set value in nm)')
  
  parser.add_argument('-boxz', dest='boxz', default=0.0, type=float,
  help='Size of z coordinate of PBC box (from 0 to set value in nm)')
  
  parser.add_argument('-nofit', dest='nofit', default='False',
  help='Disable fitting, the trajectory must be properly fited (default False)')
  
  parser.add_argument('-layers', dest='layers', default=1, type=int,
  help='Number of hidden layers (allowed values 1-3, default = 1)')
  
  parser.add_argument('-layer1', dest='layer1', default=256, type=int,
  help='Number of neurons in the first encoding layer (default = 256)')
  
  parser.add_argument('-layer2', dest='layer2', default=256, type=int,
  help='Number of neurons in the second encoding layer (default = 256)')
  
  parser.add_argument('-layer3', dest='layer3', default=256, type=int,
  help='Number of neurons in the third encoding layer (default = 256)')
  
  parser.add_argument('-actfun1', dest='actfun1', default='sigmoid',
  help='Activation function of the first layer (default = sigmoid, for options see keras documentation)')
  
  parser.add_argument('-actfun2', dest='actfun2', default='linear',
  help='Activation function of the second layer (default = linear, for options see keras documentation)')
  
  parser.add_argument('-actfun3', dest='actfun3', default='linear',
  help='Activation function of the third layer (default = linear, for options see keras documentation)')
  
  parser.add_argument('-optim', dest='optim', default='adam',
  help='Optimizer (default = adam, for options see keras documentation)')
  
  parser.add_argument('-epochs', dest='epochs', default=100, type=int,
  help='Number of epochs (default = 100, >1000 may be necessary for real life applications)')
  
  parser.add_argument('-shuffle_interval', dest='shuffle_interval', default=0, type=int,
  help='Shuffle interval (default = number of epochs + 1)')
  
  parser.add_argument('-batch', dest='batch_size', default=0, type=int,
  help='Batch size (0 = no batches, default = 0)')
  
  parser.add_argument('-lr', dest='lr', default=0.001, type=float,
  help='Learning rate (default 0.001)')
  
  parser.add_argument('-o', dest='ofile', default='',
  help='Output file with values of t-SNE embeddings (txt, default = no output)')
  
  parser.add_argument('-model', dest='modelfile', default='',
  help='Prefix for output model files (experimental, default = no output)')
  
  parser.add_argument('-plumed', dest='plumedfile', 
  help='Output file for Plumed (default = no output)')
  
  parser.add_argument('-plumed2', dest='plumedfile2',
  help='Output file for Plumed >= 2.6 (default = no output)')
  
  parser.add_argument('-plumed3', dest='plumedfile3',
  help='Output file for Plumed with PYTORCH_MODEL_CV (default - no output)')
  
  
  return parser.parse_args()

def process_args(args):
  class ddict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

  a = ddict()

  a.infilename = args.infile
  a.intopname = args.intop
  a.boxx = args.boxx
  a.boxy = args.boxy
  a.boxz = args.boxz
  a.embed_dim = args.embed_dim
  a.perplexity = args.perplex
  a.nofit = args.nofit
  if args.nofit == "True":
    a.nofit = 1
  elif args.nofit == "False":
    a.nofit = 0
  else:
    print("ERROR: -nofit %s not understood" % args.nofit)
    exit(0)
  
  if args.layers < 1 or args.layers > 3:
    print("ERROR: -layers must be 1-3, for deeper learning contact authors")
    exit(0)
  if args.layer1 > 1024:
    print("WARNING: You plan to use %i neurons in the first layer, could be slow")
  if args.layers == 2:
    if args.layer2 > 1024:
      print("WARNING: You plan to use %i neurons in the second layer, could be slow")
  if args.layers == 3:
    if args.layer3 > 1024:
      print("WARNING: You plan to use %i neurons in the third layer, could be slow")
  if args.actfun1 not in ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']:
    print("ERROR: cannot understand -actfun1 %s" % args.actfun1)
    exit(0)
  if args.layers == 2:
    if args.actfun2 not in ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']:
      print("ERROR: cannot understand -actfun2 %s" % args.actfun1)
      exit(0)
  if args.layers == 3:
    if args.actfun3 not in ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']:
      print("ERROR: cannot understand -actfun3 %s" % args.actfun3)
      exit(0)
  if args.layers == 1 and args.actfun2!='linear':
    print("ERROR: actfun2 must be linear for -layers 1")
    exit(0)
  if args.layers == 2 and args.actfun3!='linear':
    print("ERROR: actfun3 must be linear for -layers 2")
    exit(0)
  a.layers = args.layers
  a.layer1 = args.layer1
  a.layer2 = args.layer2
  a.layer3 = args.layer3
  a.actfun1 = args.actfun1
  a.actfun2 = args.actfun2
  a.actfun3 = args.actfun3
  a.optim = args.optim
  a.epochs = args.epochs
  a.shuffle_interval = args.shuffle_interval
  a.batch_size = args.batch_size
  a.lr = args.lr
  if args.ofile[-4:] == '.txt':
    a.ofilename = args.ofile
  elif len(args.ofile)>0:
    a.ofilename = args.ofile + '.txt'
  else:
    a.ofilename = ''
  a.modelfile = args.modelfile
  a.plumedfile = args.plumedfile
  if a.plumedfile and a.plumedfile[-4:] != '.dat':
    a.plumedfile = a.plumedfile + '.dat'
  a.plumedfile2 = args.plumedfile2
  if a.plumedfile2 and a.plumedfile2[-4:] != '.dat':
    a.plumedfile2 = a.plumedfile2 + '.dat'
  a.plumedfile3 = args.plumedfile3
  if a.plumedfile3 and a.plumedfile3[-4:] != '.dat':
    a.plumedfile3 = a.plumedfile3 + '.dat'
  a.fullcommand = ""
  for item in sys.argv:
    a.fullcommand = a.fullcommand + item + " "
  
  return a
