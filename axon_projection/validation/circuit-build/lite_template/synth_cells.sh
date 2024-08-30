#!/bin/bash -l
export SIM_FOLDER=$(pwd); \
export LOG_LOC=${SIM_FOLDER}/synth_cells_$(date "+%Y%m%d_%Hh%Mm%Ss").out; \

module load unstable circuit-build && \
nohup circuit-build run synthesize_morphologies --bioname ${SIM_FOLDER}/bioname --cluster-config ${SIM_FOLDER}/bioname/cluster.yaml &> ${LOG_LOC} &
