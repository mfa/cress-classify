#!/bin/sh

## prepare csv for dry/wet histogram
# for threshold in 3000 6000 8000; do
#     head -n 1 dry-wet-split--48.csv > all-dry-wet-${threshold}.csv
#     grep "|${threshold}|" dry-wet-*csv -h >> all-dry-wet-${threshold}.csv
# done

## prepare csv of cross cycle results
head -n 1 data/model_v3_85_128_dense64_threshold6000_batchsize32--cycle--71--scores.csv > data/all-cross-cycle-scores.csv
cat data/*--cycle--* | grep '^0|' >> data/all-cross-cycle-scores.csv
