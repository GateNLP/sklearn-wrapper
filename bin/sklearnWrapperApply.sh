#!/bin/bash

ROOTDIR="$1"
shift
model="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/sklearnApply.py" "${model}" $@
python "${ROOTDIR}/python/sklearnApply.py" "${model}" $@
popd >/dev/null
