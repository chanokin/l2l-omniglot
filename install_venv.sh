#!/usr/bin/env bash

JUBE=1
BASE=`pwd`
VENV=$BASE/venv3
GENN=$BASE/genn
PYNN=$BASE/pynn_genn
L2L=$BASE/L2L
SWIG_TMP=$BASE/swig_tmp
SWIG=$BASE/swig
NVCC=`which nvcc`
CUDA=`echo $NVCC | sed 's/\/bin\/nvcc//g'`
INSTALL_SWIG=1
INSTALL_JUBE=1
INSTALL_VENV=1

if [ $INSTALL_SWIG -eq 1 ]
then
  echo "Installing SWIG"
  git clone https://github.com/swig/swig $SWIG_TMP
  cd $SWIG_TMP
  git checkout tags/rel-3.0.12
  ./autogen.sh
  ./configure --prefix=$SWIG
  make
  make install
fi

echo 'Installing venv'
cd $BASE
echo $BASE
echo $VENV

if [ $INSTALL_VENV -eq 1 ]
then
  python3 -m venv $VENV
fi


printf "\nexport CUDA_PATH=%s\n" $CUDA >> $VENV/bin/activate
printf "\nexport PATH=\$PATH:%s/bin\n" $SWIG >> $VENV/bin/activate
printf "\nexport PATH=\$PATH:%s/bin\n\n" $GENN >> $VENV/bin/activate


PYVER=`ls $VENV/lib/`
ln -s $BASE/omnigloter $VENV/lib/$PYVER/site-packages/


source $VENV/bin/activate

pip install --upgrade pip
pip install Cython
pip install numpy
pip install scipy
pip install matplotlib
pip install setuptools
pip install scikit-learn
pip install skutil
pip install pyyaml

git clone https://github.com/genn-team/genn $GENN
cd $GENN

make LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make DYNAMIC=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
#make DYNAMIC=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
#make MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/

make CPU_ONLY=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make CPU_ONLY=True DYNAMIC=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
#make CPU_ONLY=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
#make CPU_ONLY=True DYNAMIC=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/

#sed -i 's/numpy>1\.6, < 1\.15/numpy>1\.6/g' setup.py
python setup.py develop

git clone https://github.com/genn-team/pynn_genn $PYNN
cd $PYNN
#sed -i 's/numpy>1\.6, < 1\.15/numpy>1\.6/g' setup.py
python setup.py develop

if [ $INSTALL_JUBE -eq 1 ]
then
    cd $BASE
    tar xf JUBE-Sandra-demonstrator.tar
    cd JUBE/JUBE
    python setup.py develop
    cd $BASE
    rm -fr LTL
fi

git clone https://github.com/chanokin/L2L $L2L
cd $L2L
if [ $JUBE -eq 1 ]
then
    ./install.sh $VENV/bin/activate
else
    python setup.py develop
fi



