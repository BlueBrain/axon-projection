#!/bin/bash -l
source inputs.sh && \
export LOG_LOC=${SIM_FOLDER}/synth_axons_$(date "+%Y%m%d_%Hh%Mm%Ss").out; \
source /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/new_venv135/bin/activate && \
export FILTERED_COLLECTION_PATH=$(python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py gfp ${SIM_FOLDER} ${AXONS_REGION} |tail -n 1) && \
echo "Synthesizing axons using collection ${FILTERED_COLLECTION_PATH}"; \
nohup /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/axon_synth.sh ${SIM_FOLDER} ${A_S_CFG} ${FILTERED_COLLECTION_PATH} ${A_P_OUT} &> ${LOG_LOC} &
