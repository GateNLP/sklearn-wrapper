from __future__ import print_function
import sys
import sklearn as skl 
from sklearn.externals import joblib
import json
import numpy as np
from flask import Flask
from flask import request
from flask import Response


## First of all, check the arguments: we expect the model prefix and the server port number
if len(sys.argv) != 3:
	sys.exit("ERROR: Not exactly three arguments: [script], model path prefix, and port number")

modelpath=sys.argv[1]
if not modelpath:
	sys.exit("ERROR: No model path")

port=int(sys.argv[2])

## get the model
model=joblib.load(modelpath)
model.probability=True
canProbs = hasattr(model,"predict_proba") and callable(getattr(model,"predict_proba"))

app = Flask(__name__)

app.debug=True

def errorResponse(string):
  resp = Response(string)
  resp.status_code = 400
  resp.headers["Content-type"] = "text/plain"
  return resp

## Convert values/indices representation to csr_matrix representation
def to_csr(values,indices):
  csrinds = []
  csrvals = []
  csrptrs = [0]
  ninst=len(values)  ## number of instances (nested vectors in values and indices)
  for i in range(ninst):
    vals = values[i]
    inds = indices[i]
    for j in range(len(vals)):
      csrinds.append(inds[j])
      csrvals.append(vals[j])
    csrptrs.append(len(csrinds))
  m=csr_matrix((csrvals,csrinds,csrptrs),dtype='float64')
  return m

## We allow POST only, but we deal with GET explicitly
@app.route("/", methods=['GET'])
def processGet():
  return errorResponse("ERROR: only POST requests are allowed on this service!\n")

@app.route("/", methods=['POST'])
def processPost():
  ct=request.headers.get("Content-Type","UNKNOWN")
  if ct is not "application/json":
    return errorResponse("ERROR: only application/json is accepted as content-type!\n")
  body=request.get_data()
  if not body:
    return errorResponse("ERROR: got an empty request body, expected JSON string\n")
  map = json.loads(body)
  ## we expect the same format as for weka: 
  ##   values: an array of double arrays with the sparse values
  ##   indices: an array of integer arrays with the sparse indices
  ##   weights: an optional array of weights, TODO: ignored for now!!
  values = map.get("values")
  indices = map.get("indices")
  weights = map.get("weights")
  if not values:
    return errorResponse("ERROR: got a JSON request without a values field\n")
  if not indices:
    return errorResponse("ERROR: got a JSON request without an indices field\n")
  ## convert our own representation to a csr_matrix 
  X=to_csr(values,indices)
  ## Apply the model
  ret = {}
  if canProbs:
		probs = model.predict_proba(X)
		ret["preds"] = probs.tolist()
  else:
  	targets = model.predict(X)
  	ret["probs"] = [[i] for i in targets]
  json=json.dumps(ret)
  resp=Response(string)
  resp.status_code=200
  resp.headers["Content-type"] = "application/json"
  return resp

## Other servers can handle stop, we just cheerfully ignore it
@app.route("/stop")
def processStop():
  return "NO ERROR: stop command ignored, not supported\n"

if __name__ == "__main__":
  app.run(host='127.0.0.1',port=port)
