FROM rayproject/ray-ml:2.5.0-gpu

USER root

RUN pip3 install mdtraj

COPY dist/parmtSNEcv-0.1.tar.gz /tmp/
RUN pip3 install /tmp/parmtSNEcv-0.1.tar.gz


USER 1000

