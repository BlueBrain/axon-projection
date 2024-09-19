#!/bin/bash
# $1 : path to folder containing bioname == sim_folder
# $2 : region to synth axons (eg "MOp")
# $3 : mtype to synth axons (eg "PC")
# $4 : path to axon-synth config file
# $5 : path to axonal-projection output (eg "/gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/out_ML_7")
source /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/new_venv135/bin/activate && \
FILTERED_COLLECTION_PATH=$(python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py gfp $1 $2 |tail -n 1)
python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py rm $1 $FILTERED_COLLECTION_PATH && \
python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py dm $1 $FILTERED_COLLECTION_PATH && \
python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/change_morph_extension.py $1 && \
echo "Done post processing axons" #&& \
# python /gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/add_axons.py ra $1 $FILTERED_COLLECTION_PATH && \
# find $1 -name "*tmp_local_axons*" | xargs rm -rf
