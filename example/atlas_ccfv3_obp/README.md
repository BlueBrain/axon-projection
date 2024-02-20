This atlas was extracted from Nexus from https://bbp.epfl.ch/neurosciencegraph/data/4906ab85-694f-469d-962f-c0174e901885
The function used for staging is here:
https://bbpgitlab.epfl.ch/nse/cwl-registry/-/blob/main/src/cwl_registry/staging.py?ref_type=heads#L403

Then the raw orientation field was updated to replace nan values by default orientation (1, 0, 0, 0).
The function used to fix the orientations is the following (extracted from
https://bbpgitlab.epfl.ch/nse/cwl-registry/-/blob/main/src/cwl_registry/mmodel/recipe.py?ref_type=heads#L82):

def build_cell_orientation_field(brain_regions, orientations=None):
    """Create a cell orientation field."""
    final = np.full(list(brain_regions.shape) + [4], fill_value=np.nan)

    in_brain = brain_regions.raw != 0

    # placeholder quaternions for all in-brain voxels
    final[in_brain] = (1.0, 0.0, 0.0, 0.0)

    # overwrite with non-nan quaternions
    if orientations:
        not_nan = in_brain & ~np.any(np.isnan(orientations.raw), axis=-1)
        final[not_nan] = orientations.raw[not_nan]

    return brain_regions.with_data(final)
