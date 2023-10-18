docker run \
	--gpus all \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-u $(id -u) -ti -v $PWD:/work -w /work \
	ljocha/ptltsne parmtSNEcv "$@"

