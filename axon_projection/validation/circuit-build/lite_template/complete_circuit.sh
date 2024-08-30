#!/bin/bash -l
export SIM_FOLDER=$(pwd); \
export LOG_LOC=${SIM_FOLDER}/complete_circuit_$(date "+%Y%m%d_%Hh%Mm%Ss").out; \

rm -rvf ${SIM_FOLDER}/connectome && \
rm -rvf ${SIM_FOLDER}/sonata/networks/edges && \
module load unstable circuit-build && \
# finally complete the circuit-build workflow
nohup circuit-build run structural --bioname ${SIM_FOLDER}/bioname --cluster-config ${SIM_FOLDER}/bioname/cluster.yaml &> ${LOG_LOC} &
