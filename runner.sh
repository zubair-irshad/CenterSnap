#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$(readlink -f "${SCRIPT_DIR}")
export PYTHONPATH=$PYTHONPATH:$(readlink -f "${SCRIPT_DIR}/export")
export OPENBLAS_NUM_THREADS=1
export PYOPENGL_PLATFORM=egl

$SCRIPT_DIR/env/bin/python $@
