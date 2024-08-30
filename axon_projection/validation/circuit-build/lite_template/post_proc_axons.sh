#!/bin/bash -l
source inputs.sh && \
export LOG_LOC=${SIM_FOLDER}/post_proc_$(date "+%Y%m%d_%Hh%Mm%Ss").out; \

nohup /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/post_proc_axon_synth.sh ${SIM_FOLDER} ${AXONS_REGION} ${MTYPE} ${A_S_CFG} ${A_P_OUT} &> ${LOG_LOC} &
