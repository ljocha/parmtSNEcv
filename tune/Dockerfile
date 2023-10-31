FROM cerit.io/ljocha/ray-ml:2.7.1-gpu

USER root

RUN pip3 install mdtraj

COPY parmtSNEcv-0.1.tar.gz /tmp/
RUN pip3 install /tmp/parmtSNEcv-0.1.tar.gz


USER 1000

