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

echo 1>&2 ${mypython} "${ROOTDIR}/python/costclaTrain.py" "${data}" "${model}" "${algorithm}" $@
${mypython} "${ROOTDIR}/python/costclaTrain.py" "${data}" "${model}" "${algorithm}" $@
popd >/dev/null
