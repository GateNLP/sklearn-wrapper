from __future__ import print_function
import sys
import sklearn as skl 
from sklearn.externals import joblib
import json
from scipy.sparse import csr_matrix
import numpy as np

## TODO make it work with regression!

##print ("sklearnApply - got args: ", sys.argv, file=sys.stderr)
if len(sys.argv) != 2:
	sys.exit("ERROR: Not exactly two arguments: [script] and model path")

modelpath=sys.argv[1]
if not modelpath:
	sys.exit("ERROR: No model path")


## get the model
model=joblib.load(modelpath)
model.probability=True
canProbs = hasattr(model,"predict_proba") and callable(getattr(model,"predict_proba"))

## Now iterate through reading json from standard input
## We expect a map which either contains data to find predictions for or
## just an indication to stop processing.
## The json object contains the following keys:
## - cmd: either "STOP" to stop processing or "CSR1" for the representation below
## - values: an array of values for creating the sparse matrix
## - rowinds: row indices of values
## - colinds: column indices of values
## - shaperows: number of rows 
## - shapecols: number of columns 
## The response gets written to standard output as a line of json with the following format
## - status: "OK" or some error message
## - targets: array of prediction values (float64)
## - probas: array of arrays of per-class probabilities 
## TODO: check situaton for regression!

nlines=0
## NOTE: apparently there is a bug in python prior to 3.3 
## that forces the use of Ctrl-D twice to get EOF from the command line!!
##print("sklearnApply: before loop",file=sys.stderr)
while True:
	line = sys.stdin.readline()
	##print("sklearnApply - got json line",file=sys.stderr)
	if line == "" :
	  break
	nlines = nlines + 1
	map=json.loads(line)
	##print "JSON parsed: ",map
	if map['cmd'] == "STOP":
		break	
	elif map['cmd'] == "CSR1":
	    X = csr_matrix((map['values'],(map['rowinds'],map['colinds'])),shape=(map['shaperows'],map['shapecols']))
	    ## print "Matrix is: ",X.toarray()
	else:
		sys.exit("ERROR invalid or no command in JSON: "+map['cmd'])

	ret = {}
	ret["status"] = "OK"
	if canProbs:
		probs = model.predict_proba(X)
		targets = np.argmax(probs,axis=1).astype("float64")
		#print "Got probs: ",probs
		#print "Got targets: ",targets
		ret["targets"] = targets.tolist()
		ret["probas"] = probs.tolist()
	else:
		targets = model.predict(X)
		#print "Got targets: ",targets
		ret["targets"] = targets.tolist()

	##print("sklearnApply: sending response",file=sys.stderr)
	print(json.dumps(ret))
	sys.stdout.flush()
	##print("sklearnApply: response sent",file=sys.stderr)

	

##print("Lines read: ", nlines,file=sys.stderr)
