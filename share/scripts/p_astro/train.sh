#!/usr/bin/env bash

# use USER_P_ASTRO_DIR env variable as base directory, else pwd.
base_dir=.
base_dir=${USER_P_ASTRO_DIR:-$base_dir}
data_dir="${base_dir}/data"
out_dir="${base_dir}/models"
log_file="${out_dir}/train_p_astro.log"

python train.py \
    ${data_dir} \
    ${out_dir} \
    --far-star 3e-4 \
    --snr-star 8.5 \
    --overwrite \
    --verbose \
    --log-file ${log_file}
