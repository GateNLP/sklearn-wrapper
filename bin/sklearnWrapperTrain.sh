#!/bin/bash

ROOTDIR="$1"
shift
data="$1"
shift
model="$1"
shift
algorithm="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/sklearnTrain.py" "${data}" "${model}" "${algorithm}" $@
python "${ROOTDIR}/python/sklearnTrain.py" "${data}" "${model}" "${algorithm}" $@
popd >/dev/null
