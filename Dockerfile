FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN pip3 install mdtraj
RUN pip3 install 'ray[tune]'
RUN pip3 install hpbandster ConfigSpace

COPY dist/parmtSNEcv-0.1.tar.gz /tmp/
RUN pip3 install /tmp/parmtSNEcv-0.1.tar.gz



