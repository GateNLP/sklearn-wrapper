from __future__ import print_function
import sys
import sklearn as skl 
import importlib
import scipy.io as sio
from sklearn.externals import joblib
import os
import inspect

## IMPORTANT: the option values need to be valid pythong expressions, and they
## will get evaluated! So in order to pass a string, enclose the value in quotes!

## NOTE: some algorithms have the n_jobs attribute which can be used to enable 
## parallel processing!

## NOTE: if a instweights.mtx file is present, the weights option of fit is used!

print("costclaTrain - got args: ", sys.argv, file=sys.stderr)
if len(sys.argv) < 4:
	sys.exit("ERROR: Not at least four arguments: [script], data base name, model base name, algorithm name and options")

data=sys.argv[1]
if not data:
	sys.exit("ERROR: No data path")

modelpath=sys.argv[2]
if not modelpath:
	sys.exit("ERROR: No model path")

alg=sys.argv[3]
if not alg:
	sys.exit("ERROR: No algorithm name")

options=sys.argv[4:]
## check that if there are options, they come in pairs!
if len(options) % 2 != 0:
        print("ERROR: need even number of name/value arguments for the options: got ",options,file=sys.stderr);
	sys.exit("ERROR: sklearnTrain aborted")

## prepare the training algorithm
## NOTE: in Python 3 we should use functions from the imp module?
tmp = alg.rsplit('.',1)
if not len(tmp) == 2:
	sys.exit("ERROR: no dot in algorithm name: "+alg)

m, c = tmp
clazz = getattr(importlib.import_module(m),c)
model = clazz()
## check that we actually can  use the costs
canCosts = "cost_mat" in inspect.getargspec(model.fit).args
if not canCosts:
	sys.exit("ERROR: model does not support cost_mat")

## set our own default options
## model.probability=True
model.verbose = True
## to make things easy, allow the attrname to start with a hyphen or not
## Process option pairs/
for i in range(0, len(options), 2):
	optname = options[i]
	optval = eval(options[i+1])
	if optname.startswith("-"):
		optname = optname[1:]
	## try to set the attribute: check first if the attribute exists
	if hasattr(model,optname):
	    setattr(model,optname,optval)
	else:
		sys.exit("ERROR: cannot use option "+optname)

## load the data: we expect two files in Matrix 
## The parameter is the prefix to which we add "dep.mtx" and "indep.mtx" to get the final names
depfile = data+"dep.mtx"
indepfile=data+"indep.mtx"
weightsfile=data+"instweights.mtx"
costsfile=data+"instcosts.mtx"


deps = sio.mmread(depfile)
indeps = sio.mmread(indepfile)
deps = deps.toarray().reshape(deps.shape[0],)
costs = sio.mmread(costsfile)
costs = costs.toarray()

canWeights = "sample_weight" in inspect.getargspec(model.fit).args

if canWeights and os.path.isfile(weightsfile):
	weights = sio.mmread(weightsfile)
	weights = weights.toarray().reshape(weights.shape[0],)	
	model.fit(indeps,deps,sample_weight=weights,cost_mat=costs)
else:	
    model.fit(indeps,deps,cost_mat=costs)

joblib.dump(model,modelpath)
