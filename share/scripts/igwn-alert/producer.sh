#!/usr/bin/env bash

log_file="./producer.log"

python spiir_producer_test.py \
    --pipeline "spiir" \
    --group "Test" \
    --service-url "https://gracedb-playground.ligo.org/api/" \
    --log-file $log_file \
    --debug \
    "../../data/pipeline/coinc/H1L1_1257994429_387_373.xml" \
    "../../data/pipeline/coinc/H1L1_1258008860_384_184.xml" \
    "../../data/pipeline/coinc//H1L1_1258020280_386_186.xml" \
    "../../data/pipeline/coinc//H1L1_1258077746_385_772.xml" \
    "../../data/pipeline/coinc//H1L1_1257995031_386_213.xml" \
    "../../data/pipeline/coinc//H1L1_1258009986_386_526.xml" \
    "../../data/pipeline/coinc//H1L1_1258021189_389_643.xml" \
    "../../data/pipeline/coinc//H1L1_1258091887_386_845.xml" \
    "../../data/pipeline/coinc//H1L1_1257997458_384_98.xml" \
    "../../data/pipeline/coinc//H1L1_1258013401_385_497.xml" \
    "../../data/pipeline/coinc//H1L1_1258035632_387_81.xml"
