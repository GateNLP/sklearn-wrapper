from __future__ import print_function
import sys
import sklearn as skl 
import importlib
import scipy.io as sio
from sklearn.externals import joblib
import os
import inspect
import numpy as np
import pickle

## IMPORTANT: costcla algorithms only support binary classification. This 
## means that the cost matrix we get must contain two costs so the shape
## of what we get must be (n,2)
## However, the algorithm expects 4 values: from the full confusion matrix:
## if the matrix is this:
## 
##             actual c0   actual c1
##   pred c0      C-tn       c-fn
##   pred c1      c-fp       c-tp
##
## Then for each instance the algorithm expects a vector 
##    c-fp, c-fn, c-tp, c-tn  (according to private communication)
##
## We only get the two costs: assigning class 0 and assigning class 1: [c0, c1]
## There are two cases:
##    c0 < c1: true class is 0 so c0 is cTN and c1 is is cFP -- for this example,
##          the true class is 0 so there is no TP -- not sure which cost to assign?
##          same for FN: there is not FN so what is the cost?
##    [c0, c1] => [c1, ?, ?, c0] => [c1, c1, c0, c0]
##    else:  true class is 1 so c0 is cFN and c1 is cTP again the other two do not exist?
##    [c0, c1] => [?, c0, c1, ?] => [c0, c0, c1, c1]

## IMPORTANT: the option values need to be valid python expressions, and they
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
costs = sio.mmread(costsfile).toarray()
print("Costs shape after reading: ",costs.shape, file=sys.stderr)
print("Costs e0 after reading: ",costs[0], type(costs[0]), costs[0][0], file=sys.stderr)

deps = deps.toarray().reshape(deps.shape[0],)
indeps = indeps.toarray()

def conv1(x):
    if(x[0]<x[1]):
        return [x[1],x[1],x[0],x[0]]
    else:
        return [x[0],x[0],x[1],x[1]]

def conv2(x):
    if(x[0]<x[1]):
        return [x[1],x[1],x[1],x[0]]
    else:
        return [x[0],x[0],x[0],x[1]]

tmp=list(map(conv1,costs))
## print("tmp: ",tmp,file=sys.stderr)
costs=np.array(tmp)
print("Costs shape after conversion: ",costs.shape, file=sys.stderr)

canWeights = "sample_weight" in inspect.getargspec(model.fit).args

if canWeights and os.path.isfile(weightsfile):
	weights = sio.mmread(weightsfile)
	weights = weights.toarray().reshape(weights.shape[0],)	
	model.fit(indeps,deps,sample_weight=weights,cost_mat=costs)
else:
	print("indeps shape: ",indeps.shape,file=sys.stderr)
	print("deps shape: ",deps.shape,file=sys.stderr)
	print("costs shape: ",costs.shape,file=sys.stderr)
	print("Trying to fit model: ",model,file=sys.stderr)
	model.fit(indeps,deps,cost_mat=costs)

## joblib dump does not seem to work
## joblib.dump(model,modelpath)

pickle.dump(model,open(modelpath,'wb'))
