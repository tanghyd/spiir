#!/usr/bin/env bash

username=

signal_config="../p_astro/models/fgmc.pkl"
source_config="../p_astro/models/mchirp_area.pkl"
log_file="./consumer.log"

python spiir_consumer_test.py \
    $signal_config \
    $source_config \
    --out "./out" \
    --topics "test_spiir" \
    --group "gracedb-playground" \
    --username $username \
    --upload \
    --save-payload \
    --log-file $log_file \
    --debug
