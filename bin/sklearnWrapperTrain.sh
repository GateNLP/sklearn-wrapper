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

python "${ROOTDIR}/python/sklearnTrain.py" "${data}" "${model}" "${algorithm}" $@
popd >/dev/null
