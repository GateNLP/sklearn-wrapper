#!/bin/bash

ROOTDIR="$1"
shift
model="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/costclaApply.py" "${model}" $@
python3 "${ROOTDIR}/python/costclaApply.py" "${model}" $@
popd >/dev/null
