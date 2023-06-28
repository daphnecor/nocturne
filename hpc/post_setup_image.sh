DEV_MODE=true
PROJECT="nocturne-research"
NOCTURNE_DIR="/scratch/$USER/nocturne"

#module load /scratch/$USER/openmpi/intel/4.0.5
module load openmpi/intel/4.0.5

if [ -d "$NOCTURNE_DIR" ]; then

    # Install requirements if available
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt --no-cache-dir
    fi

    # Install development requirements if DEV_MODE and available
    if [ $DEV_MODE ] && [ -f requirements.dev.txt ]; then
        pip install -r requirements.dev.txt --no-cache-dir
    fi

    # Install nocturne
    CWD=`pwd`
    cd $NOCTURNE_DIR
    git submodule sync
    git submodule update --init --recursive
    python setup.py develop
    cd $CWD

else
    echo "Please clone nocturne from Github to ${NOCTURNE_DIR}."
fi