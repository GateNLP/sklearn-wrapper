#!/bin/bash

ROOTDIR="$1"
shift
model="$1"
shift

pushd "$ROOTDIR" >/dev/null

python "${ROOTDIR}/python/sklearnApply.py" "${model}" $@
popd >/dev/null
