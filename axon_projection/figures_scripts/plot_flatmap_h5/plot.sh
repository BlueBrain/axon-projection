#!/bin/bash
#$1 : /gpfs/bbp.cscs.ch/project/proj82/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/a_s_out_lite/Morphologies/hashed/0f/0f7f991474f969ca8eaf85dcd9b68567.h5

echo "INFO: Expecting a morphology file as first argument"
python flatplot.py --dots --h5morpho axon --color '#000000' -p 512 --dual /gpfs/bbp.cscs.ch/project/proj82/home/bolanos/flatmap_both.nrrd $1 morpho_axon && \
python flatplot.py --dots --h5morpho soma --color '#FFFFFF' --spread 2 -p 512 --dual /gpfs/bbp.cscs.ch/project/proj82/home/bolanos/flatmap_both.nrrd $1 morpho_soma && \
convert background.png morpho_axon.png -composite temp.png && convert temp.png morpho_soma.png -composite axon_in_map.png && rm temp.png
