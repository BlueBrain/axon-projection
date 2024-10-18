mkdir -p out_synthesis_example && \
# first create a CellCollection that contains the morphologies for which we want synthesized axons
python create_collection.py && \
# and then synthesize the axons for the cells
axon-synthesis \
--no-debug \
--log-level info \
-c config_synthesis_example.cfg \
synthesize \
--morphology-dir data/morphologies_for_synthesis \
--morphology-data-file data/synthesis_collection.h5 \
--population-probabilities-file out_clustering_example/clustering_output_AS.csv \
--projection-probabilities-file out_clustering_example/conn_probs.csv \
--tuft-properties-file out_clustering_example/tuft_properties.json \
--trunk-properties-file out_clustering_example/Clustering/trunk_properties.json \
--population-tuft-number-file out_clustering_example/tufts_numbers_distribution.json \
--input-dir out_clustering_example \
--output-dir out_synthesis_example \
# --use-mpi \
# to use mpi, install axon-synthesis with pip install axon-synthesis[mpi]
