SINGULARITY_IMAGE=nocturne_latest.sif
OVERLAY_LOC=/scratch/work/public/overlay-fs-ext3
OVERLAY_FILE=overlay-15GB-500K.ext3

PROJECT=nocturne
PROJECT_DOCKER=docker://dejongmathijs/nocturne

if [ ! -f "${SINGULARITY_IMAGE}" ]; then
    echo "Pulling Docker container from ${PROJECT_DOCKER}"
    singularity pull "${PROJECT_DOCKER}"
fi

# Copy and unzip overlay file if not already done
if [ -f "${OVERLAY_FILE}" ]; then
    # Launch singularity
    echo 'Launching singularity image in OVERLAY (use) mode...'

    # Launch singularity image
    singularity exec --nv --overlay "${OVERLAY_FILE}:ro" \
        "${SINGULARITY_IMAGE}" \
        /bin/bash
else
    echo "Setting up ${BASE_IMAGE_FILE} with initial overlay ${OVERLAY_FILE}.gz"

    if [ ! -f "${OVERLAY_FILE}.gz" ]; then
        echo "Copying overlay ${OVERLAY_FILE}.gz from ${OVERLAY_LOC}..."
        cp -rp "${OVERLAY_LOC}/${OVERLAY_FILE}.gz" . -n
        echo "Unzipping overlay ${OVERLAY_FILE}.gz..."
        gunzip "${OVERLAY_FILE}.gz" -n
    fi

    # Launch singularity
    echo 'Launching singularity image in WRITE (edit) mode...'

    # Welcome message
    echo "Run 'bash ./hpc/post_setup_image.sh' to initialize ${PROJECT}, then"
    echo "  (1) create a virtual Python environment: 'python3 -m venv venv'"
    echo "  (2) activate venv: 'source venv/bin/activate'"
    echo "  (3) install nocturne: 'bash ./hpc/post_setup_image.sh'"

    # Launch singularity image
    singularity exec --nv --overlay "${OVERLAY_FILE}:rw" \
        "${SINGULARITY_IMAGE}" \
        /bin/bash
fi
