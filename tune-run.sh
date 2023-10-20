#!/bin/bash

image=ljocha/ptltsne
cmd=parmtSNEcv
if [ $(basename $0) = run-tune.sh ]; then
	cmd = tune/tune.py
fi

docker run \
	--gpus all \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-u $(id -u) -ti -v $PWD:/work -w /work \
	$image $cmd "$@"

