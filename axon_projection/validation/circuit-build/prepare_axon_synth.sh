#!/bin/bash
# $1 : path to folder containing bioname == sim_folder
# $2 : regions to synth axons (eg "['MOp']" or "['Isocortex', 'SPVI']")
# $3 : mtype to synth axons (eg "PC")
# $4 : path to axon-synth config file
# $5 : path to axonal-projection output (eg "/gpfs/bbp.cscs.ch/project/proj82/home/petkantc/axon/axonal-projection/axon_projection/out_ML_7")
source /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/new_venv135/bin/activate && \
FILTERED_COLLECTION_PATH=$(python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py fc $1 $2 $3 "$4/clustering_output_AS.csv" |tail -n 1) && \
echo "Filtered collection path: $FILTERED_COLLECTION_PATH" && \
python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py da $1 $FILTERED_COLLECTION_PATH h5 && \
python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py da $1 $FILTERED_COLLECTION_PATH asc && \
python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py da $1 $FILTERED_COLLECTION_PATH swc && \
echo "Done preparing axons"
