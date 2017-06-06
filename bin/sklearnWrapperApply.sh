#!/bin/bash

ROOTDIR="$1"
shift
model="$1"
shift

mypython=${SKLEARN_WRAPPER_PYTHON:-python}

pushd "$ROOTDIR" >/dev/null

echo 1>&2 ${mypython} "${ROOTDIR}/python/sklearnApply.py" "${model}" $@
${mypython} "${ROOTDIR}/python/sklearnApply.py" "${model}" $@
popd >/dev/null
