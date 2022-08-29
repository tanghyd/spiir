SOURCES=${PWD}/tmp
mkdir -p $SOURCES

PREFIX=${PWD}/include
mkdir -p $PREFIX

# GSL
p=gsl-2.6
if [ ! -d ${SOURCES}/$p ]; then
    echo -e "\\n\\n>> [`date`] Building $p"
    wget $wget_opts ftp://ftp.fu-berlin.de/unix/gnu/gsl/$p.tar.gz
    tar -xzf $p.tar.gz -C ${SOURCES}
    rm $p.tar.gz
else
    echo -e "\\n\\n>> [`date`] $p already extracted. Continuing..."
fi

pushd ${SOURCES}/$p
./configure --prefix=${PREFIX}
make -j 4
make install
popd
rm -rf ${SOURCES}/$p

# lalsuite cannot find GSL
export LD_LIBRARY_PATH=${PREFIX}:${LD_LIBRARY_PATH}

# LALSuite
p=lalsuite
if [ ! -d ${SOURCES}/$p ]; then
    echo -e "\\n\\n>> [`date`] Cloning $p"
    LIGO_GIT=https://git.ligo.org/lscsoft
    git clone $LIGO_GIT/$p.git ${SOURCES}/$p
else
    echo -e "\\n\\n>> [`date`] $p already cloned. Continuing..."
fi

pushd ${SOURCES}/$p
./00boot
./configure --prefix=${PREFIX}
make -j 4
make install
popd
# rm -rf ${SOURCES}/$p