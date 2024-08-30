#!/bin/bash -l
source inputs.sh && \
export LOG_LOC=${SIM_FOLDER}/prepare_axons_$(date "+%Y%m%d_%Hh%Mm%Ss").out; \
# $1 : path to folder containing bioname == sim_folder
# $2 : region to synth axons (eg "MOp")
# $3 : mtype to synth axons (eg "PC")
# $4 : path to axon-synth config file
# $5 : path to axonal-projection output (eg "/gpfs/bbp.cscs.ch/project/proj82/home/petkantc/axon/axonal-projection/axon_projection/out_ML_7")
# $6 : run region-grower (true/false)
nohup /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/prepare_axon_synth.sh ${SIM_FOLDER} ${AXONS_REGION} ${MTYPE} ${A_P_OUT} &> ${LOG_LOC} &
