#!/bin/bash

true ${image:=ljocha/ptltsne}

cmd=parmtSNEcv
gpuflags="--gpus all"

if [ $(basename $0) = tune-run.sh ]; then
	cmd=tune/tune.py
elif [ $(basename $0) = tune-run-amd.sh ]; then
	cmd=tune/tune.py
	image=$image:amd
	gpuflags="--device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render"
elif [ $(basename $0) = devel-run-amd.sh ]; then
	image=$image:amd
	gpuflags="--device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render"
fi

docker run --rm \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-u $(id -u) -ti -v $PWD:/work -w /work \
	$gpuflags $image $cmd "$@"

