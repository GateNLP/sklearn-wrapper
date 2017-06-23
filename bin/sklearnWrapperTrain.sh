#!/bin/bash

ROOTDIR="$SKLEARN_WRAPPER_HOME"
data="$1"
shift
model="$1"
shift
algorithm="$1"
shift

mypython=${SKLEARN_WRAPPER_PYTHON:-python}

pushd "$ROOTDIR" >/dev/null

echo 1>&2 ${mypython} "${ROOTDIR}/python/sklearnTrain.py" "${data}" "${model}" "${algorithm}" $@
${mypython3} "${ROOTDIR}/python/sklearnTrain.py" "${data}" "${model}" "${algorithm}" $@
popd >/dev/null
