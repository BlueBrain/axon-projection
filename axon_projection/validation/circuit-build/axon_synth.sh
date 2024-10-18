#!/bin/bash
# $1 : path to sim_folder (eg "lite")
# $2 : path to synth config file
# $3 : name of cell_collection file of morphs to synthesize axons
# $4 : path to axon-projection out folder
deactivate > /dev/null; \
module purge; \
module load archive/2024-01 py-mpi4py; \
module load archive/2024-01 py-dask-mpi; \
\
export CURRENT_TAG=$(date "+%Y%m%d_%Hh%Mm%Ss"); \
export ROOT_DIR=/gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build; \
export OUTPUT_DIR=$1/a_s_out; \
export LOG_FILE=$1/log_${CURRENT_TAG}.txt; \
# source ${ROOT_DIR}/venv/bin/activate; \
source /gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/venvAxonSynth/bin/activate; \
echo "Results exported to ${OUTPUT_DIR}"; \
# echo "Logs exported to ${LOG_FILE}"; \
# echo "Module list:"; \
# module list 2>&1 | tee -a ${LOG_FILE}; \
# echo "Pip freeze:"; \
# pip freeze 2>&1 | tee -a ${LOG_FILE}; \
\
stdbuf -oL -eL \
salloc \
        --partition=prod \
        --constraint=cpu \
        --account=proj135 \
        --exclusive \
        --time=10:00:00 \
        -n200 \
        --cpus-per-task=1 \
        --mem=0 \
        --job-name=axon-synth \
        srun \
                axon-synthesis \
                --no-debug \
                --log-level info \
                -c $2 \
                synthesize \
                --morphology-dir $1/morphologies/neurons \
                --morphology-data-file $3 \
                --population-probabilities-file $4/clustering_output_AS.csv \
                --projection-probabilities-file $4/conn_probs.csv \
                --tuft-properties-file $4/tuft_properties.json \
                --trunk-properties-file $4/Clustering/trunk_properties.json \
                --population-tuft-number-file $4/tufts_numbers_distribution.json \
                --input-dir /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/inputs \
                --output-dir ${OUTPUT_DIR} \
                --outputs-disable-final-figures \
                --outputs-disable-graph-creation-figures \
                --outputs-disable-graph-creation-data \
                --outputs-disable-main-trunk-figures \
                --outputs-disable-main-trunk-morphologies \
                --outputs-disable-postprocess-trunk-figures \
                --outputs-disable-postprocess-trunk-morphologies \
                --outputs-disable-steiner-tree-solutions \
                --outputs-disable-target-point-figures \
                --outputs-disable-target-points \
                --outputs-disable-tuft-figures \
                --outputs-disable-tuft-morphologies \
                --use-mpi \
# 2>&1 | tee -a ${LOG_FILE}; \
echo "Results exported to ${OUTPUT_DIR}"; \
# echo "Logs exported to ${LOG_FILE}";
