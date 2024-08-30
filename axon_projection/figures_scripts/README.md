# Figures
Scripts used for generating some of the figures in the article Petkantchin et al.

TODO add DOI

## Figure 3: Clustered vs. sampled presubiculum axons

- Ran the complete `axonal_projections` workflow with the MouseLight morphologies dataset, and imposing 5 clusters to the *PRE* region, by setting `n_components = {'PRE': 5}`in `[classify]` in the configuration file.
- Ran `../plot_results.py` with `verify=True`

## Figure 4
### Flat map plot (A)
Used [plot_multiple_morphs.py](plot_flatmap_h5/plot_multiple_morphs.py) with the MOp5 morphologies of the input dataset, and the synthesized ones
### Morphometrical comparison of tufts and trunks (C, D)
Used [compute_scores.py](compute_scores.py) on the tufts and trunks of the biological and synthesized axons, from the MOp5 region.
### Lengths in regions (E, F)
Used [plot_feat_cmp.py](plot_feat_cmp.py) on the biological and synthesized axons of the MOp5 region.
## Figure 5
Used [plot_feat_cmp.py](plot_feat_cmp.py) on the biological and synthesized axons' tufts distributions of the MOp5 region.
## Figure 6
Used [build_connectome.py](build_connectome.py).
## Figure 7
Used [build_connectome.py](build_connectome.py).
