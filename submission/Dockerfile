# Bite2Text submission. CPU-only by design: the pipeline is deterministic geometry plus a
# small gradient-boosting model, so it needs no accelerator and avoids CUDA image risk.
#
# Python 3.13 deliberately matches the interpreter the model bundle was fitted under. joblib
# pickles carry the scikit-learn version they were written with, so the runtime must line up;
# the pinned numpy/scipy in requirements.txt are also 3.12+ only.
FROM --platform=linux/amd64 docker.io/library/python:3.13-slim AS bite2text_algorithm

ENV PYTHONUNBUFFERED=1

# Single-threaded numerics: inference is one row at a time, and OpenMP thread setup on a
# many-core host costs ~1000x more than the prediction itself. inference.py sets these too,
# in case the platform overrides the image environment.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

LABEL gc_upload_version="odin2026-bite2text-geometry-v1"

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY --chown=user:user src/bite2text /opt/app/bite2text
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py /opt/app/

USER user
ENV PYTHONPATH=/opt/app

ENTRYPOINT ["python", "inference.py"]
