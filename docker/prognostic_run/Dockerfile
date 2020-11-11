# syntax=docker/dockerfile:experimental
FROM us.gcr.io/vcm-ml/fms-build AS bld

ENV FMS_DIR=/FMS
ENV ESMF_DIR=/usr/local/esmf
ENV FV3GFS_FORTRAN_DIR=/external/fv3gfs-fortran
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

COPY --from=us.gcr.io/vcm-ml/esmf-build /usr/local/esmf ${ESMF_DIR}
COPY --from=us.gcr.io/vcm-ml/esmf-build /usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* ${ESMF_LIB}/

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get install -y  --no-install-recommends \
    cmake \
    python3 \
    libpython3-dev \
    python3-dev \
    python3-setuptools \
    python3-pip \
    python3-cffi \
    cython3


# install pfunit
RUN cd /tmp && curl -L https://github.com/Goddard-Fortran-Ecosystem/pFUnit/archive/3.2.9.tar.gz | tar xz
ENV F90=gfortran
ENV F90_VENDOR=GNU
ENV PFUNIT=/usr/local
RUN ls /tmp/pFUnit-3.2.9
RUN cd /tmp/pFUnit-3.2.9 && \
    cmake . && \
    make &&\
    make install INSTALL_DIR=${PFUNIT}

# install call-py-fort
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY external/fv3gfs-fortran/FV3/call_py_fort /opt/call_py_fort
ENV PYTHONPATH=/opt/call_py_fort/src/:/opt/call_py_fort/test:$PYTHONPATH
ENV CALLPY=/opt/call_py_fort
RUN cd /opt/call_py_fort/ && make && make install


# build/install the fortran model
COPY external/fv3gfs-fortran/ /tmp/fortran-build
RUN cd /tmp/fortran-build/FV3 && \
    make -j 8 && \
    PREFIX=/usr/local make install

COPY external/fv3config /tmp/fv3config
RUN pip3 install wheel
RUN pip3 install /tmp/fv3config numpy


ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG