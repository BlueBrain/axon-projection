# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Classify the axons for each source, based on the axonal projections table."""

import ast
import configparser
import logging
import sys
from os import makedirs

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from axon_projection.query_atlas import region_hemisphere
from axon_projection.query_atlas import without_hemisphere


def gmm_bic_score(estimator, data):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to *maximize*
    return -estimator.bic(data)


# Need to disable "unsubpscriptable object" error with pylint, this is a bug from their side
# pylint: disable=unsubscriptable-object,unsupported-assignment-operation
def find_good_gmm(data, n_components_range, seed=None, best=False, n_jobs=12):
    """Finds a good number of components and variance type for a GMM on the given data.

    Takes in input the data on which we want to build a Gaussian Mixture Model (GMM),
    and the maximum number of mixture components desired. Returns the best number of
    components and covariance type to use, by minimizing the BIC score on the data.

    Args:
        data (list numpy array): the feature vectors on which to base the GMM.
        In this case, we are using number of terminal points for each target region.
        n_components_range (list int) : mininmum, maximum number of mixtures to use
        (i.e. number of Gaussians).
        The algorithm will use at most n_components_max classes for the clustering.
        seed (int) : random seed, for reproducibility. Seed is used for the random initialization
            of the GMMs (as is done in K-means algorithm)
        best (bool) : if True, the best GMM found is returned. Otherwise, we pick one
            based on its BIC score.
        n_jobs (int) : number of processes used in the search.

    Returns:
        grid_search.best_params (dict) : parameters of the best GMM.
        res (pandas dataframe) : output of the search.
    """
    # give the range of number of classes to try, and allowed variances types
    param_grid = {
        "n_components": range(n_components_range[0], n_components_range[1]),
        # "covariance_type": ["spherical", "tied", "diag", "full"],
        "covariance_type": ["spherical"],
    }
    # initialize the search parameters
    grid_search = GridSearchCV(
        GaussianMixture(random_state=seed),
        param_grid=param_grid,
        scoring=gmm_bic_score,
        verbose=0,
        n_jobs=n_jobs,
    )
    # perform the search
    grid_search.fit(data)
    # extract the results in a dataframe
    res = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    # make BIC score positive because GridSearchCV expects a score to *maximize*
    res["mean_test_score"] = -res["mean_test_score"]
    # finally, rename the columns
    res = res.rename(
        columns={
            "param_n_components": "n_components",
            "param_covariance_type": "covariance_type",
            "mean_test_score": "BIC",
        }
    )

    # if we don't ask for the best, compute a pick probability for each combination of params
    min_bic = res["BIC"].min()
    # compute the probability as a Gaussian distribution
    res["pick_prob"] = np.exp(-pow(res["BIC"], 2.0) / pow(min_bic, 2.0))
    res["pick_prob"] = res["pick_prob"] / res["pick_prob"].sum()

    if best:
        logging.info("Grid search results: %s", res)
        return grid_search.best_params_, res
    # and draw a combination based on this probability
    good_params = res.sample(n=1, weights="pick_prob", random_state=seed)
    logging.info("Picked %s", good_params)
    return good_params.to_dict("records")[0], res


def compute_posteriors(morphs_df, source, gmm, source_region_id):
    """Computes probabilities for each morph to belong to each class, within a source region.

    (probabilities <=> "soft" clustering)
    """
    # get the feature vector for each morph
    feature_vectors = morphs_df.iloc[:, 3:].values
    # predict "softly" the class for each point by giving a probability to belong to each class
    class_assignment = gmm.predict(feature_vectors)
    # get the probability to belong to each class
    probas = gmm.predict_proba(feature_vectors)
    log_likelihood = gmm.score_samples(feature_vectors)
    # build the list of population ids, one per morph
    pop_ids_list = [
        str(source_region_id) + "_" + region_hemisphere(source) + "_" + str(x)
        for x in class_assignment
    ]

    # create DataFrame for morphologies, probabilities, and source region
    df_post = pd.DataFrame(
        {
            "morph_path": morphs_df["morph_path"],
            "source_region": source,  # this value will be repeated for all rows,
            "probabilities": probas.tolist(),  # convert probabilities to list
            "class_assignment": class_assignment,  # the class ID assigned to this morpho
            "population_id": pop_ids_list,  # the pop_id of this morpho
            "log_likelihood": log_likelihood.tolist(),  # prob to get each point, knowing the gmm
        }
    )
    return df_post


def run_classification(config, verify=False):
    """Creates the projectional classes for each source region.

    Inputs:
        Needs a file 'axon_{terminals|lengths}.csv' that contains feature data for classification.
        verify (bool) : if True, the clustering is run on data that was generated with the GMM.

    Outputs:
        clustering_output.csv: the GMM parameters for each class (probability to belong to
        class, means, variances), with number of points in class.
        conn_probs.csv: the probability of connection to target regions for each class.
        This probability is computed as the mean number of terminal for that target region
        and class, divided by the total number of terminal points for that class.
    """
    if not verify:
        out_path = config["output"]["path"]
    else:
        out_path = config["output"]["path"] + "verify_GMM/"

    data_path = (
        out_path
        + config["classify"]["feature_vectors_file"]
        + "_"
        + config["morphologies"]["hierarchy_level"]
        + ".csv"
    )
    makedirs(out_path + "/grid_search", exist_ok=True)
    # load feature data containing source region ("source") and number of terminals
    # in each target region ("<region_acronym>")
    f = pd.read_csv(data_path)
    if verify:
        # if we want to verify the clustering, we have to drop the class_id generated by the GMM
        # and cluster again the data that was generated with the GMM
        f.drop(["class_id"], axis=1, inplace=True)
    # first three columns are row_id, morph_path and source region, the rest is target regions
    target_region_acronyms = f.columns[3:]
    # for reproducibility
    seed = int(config["random"]["seed"])
    # this might not be needed at the moment
    np.random.seed = seed
    sources = f.source.unique()  # array of all source regions found in data
    clustering_output = []
    connexion_prob = []
    # df containing region names and acronyms, so we don't have to reload the atlas
    region_names_df = pd.read_csv(config["output"]["path"] + "region_names_df.csv", index_col=0)
    n_jobs = int(config["classify"]["n_jobs"])
    # dict of imposed number of components for each source region acronym.
    # if empty, the number of components is determined by the grid search
    imposed_n_components = ast.literal_eval(config["classify"]["n_components"])
    logging.debug("imposed_n_components: %s", imposed_n_components)
    # dataframe that will contain the (posterior) probs to belong to each class for every morph
    df_post = pd.DataFrame(
        None,
        columns=[
            "morph_path",
            "source_region",
            "probabilities",
            "class_assignment",
            "population_id",
            "log_likelihood",
        ],
    )
    # do the clustering for each source
    for s_a in sources:
        logging.info("Source region : %s", s_a)
        # dataframe with source s_a
        f_as = f[f["source"] == s_a]

        logging.info("Shape of data %s", f_as.shape)
        n_rows, _ = f_as.shape
        # skip for now source regions that have less than five data points,
        # because the data is split in five during the grid_search for the best number of mixtures
        if n_rows < 5:
            logging.info("Skip source region %s with %s point(s).", s_a, n_rows)
            # group them in the same class
            df_post = pd.concat(
                [
                    df_post,
                    pd.DataFrame(
                        {
                            "morph_path": f_as["morph_path"],
                            "source_region": s_a,  # this value will be repeated for all rows,
                            "probabilities": [1] * n_rows,  # convert probabilities to list
                            "class_assignment": [0]
                            * n_rows,  # the class ID assigned a posteriori to this morpho
                            "population_id": [
                                str(region_names_df.at[without_hemisphere(s_a), "id"]) + "_0"
                            ]
                            * n_rows,  # the population_id of the morpho
                            "log_likelihood": [0]
                            * n_rows,  # the prob to observe each point knowing the gmm
                        }
                    ),
                ]
            )
            continue
        # exclude row id, morph_path and source region in data
        data = np.array(f_as.iloc[:, 3:].values.tolist())

        # if n_components is imposed by the user for this source region, use it
        if imposed_n_components.get(s_a) is not None:  # and not verify:
            logging.info("Imposed number of components for %s : %s", s_a, imposed_n_components[s_a])
            n_components = imposed_n_components[s_a]
            cov_type = "full"
        # else, perform search for a good number of components
        else:
            n_min_components = int(n_rows / 2.0)
            n_max_components = int(n_rows * 4.0 / 5.0)  # 1/5 for cross-validation
            # search for parameters combinations that give a good BIC score
            good_params, grid_search_res = find_good_gmm(
                data,
                [n_min_components, n_max_components],
                best=(config["classify"]["best_BIC"] == "True"),
                seed=seed,
                n_jobs=n_jobs,
            )
            grid_search_res.to_markdown(out_path + "grid_search/" + s_a.replace("/", "-") + ".md")
            logging.info(good_params)

            n_components = good_params["n_components"]
            cov_type = good_params["covariance_type"]

        # we take at most n_components components for the mixture, but since it is a BayesianGM,
        # it should use the optimal number of components <= n_components
        # gmm = BayesianGaussianMixture
        # (n_components=n_rows, verbose=1, random_state=np.random.seed).fit(data)

        gmm = GaussianMixture(
            n_components=n_components, covariance_type=cov_type, verbose=1, random_state=seed
        ).fit(data)

        logging.info("GMM params: %s", gmm.get_params())
        source_brain_reg_id = region_names_df.at[without_hemisphere(s_a), "id"]
        for c in range(n_components):
            source_pop_id = str(source_brain_reg_id) + "_" + region_hemisphere(s_a) + "_" + str(c)
            clustering_output.append(
                [
                    s_a,
                    source_brain_reg_id,
                    region_hemisphere(s_a),
                    c,
                    source_pop_id,
                    int(round(n_rows * gmm.weights_[c])),
                    gmm.weights_[c],
                    gmm.means_[c].tolist(),
                    gmm.covariances_[c].tolist(),
                    gmm.get_params(),
                ]
            )
            for t in range(len(gmm.means_[c])):
                # don't show probability if it's 0
                if gmm.means_[c][t] < 1e-16:
                    continue

                target_region_id = region_names_df.at[
                    without_hemisphere(target_region_acronyms[t]), "id"
                ]
                connexion_prob.append(
                    [
                        s_a,
                        source_brain_reg_id,
                        region_hemisphere(s_a),
                        c,
                        source_pop_id,
                        target_region_acronyms[t],
                        target_region_id,
                        source_pop_id
                        + "_"
                        + str(target_region_id)
                        + "_"
                        + region_hemisphere(target_region_acronyms[t]),
                        region_hemisphere(target_region_acronyms[t]),
                        gmm.means_[c][t] / np.sum(gmm.means_[c]),
                        gmm.means_[c][t],
                    ]
                )

        # compute the probability to belong to each class for each morph
        df_post = pd.concat([df_post, compute_posteriors(f_as, s_a, gmm, source_brain_reg_id)])

    df_out = pd.DataFrame(
        clustering_output,
        columns=[
            "source",
            "brain_region_id",
            "hemisphere",
            "class_id",
            "population_id",
            "num_data_points",
            "probability",
            "means",
            "variances",
            "gmm_params",
        ],
    )
    df_out.to_csv(out_path + "clustering_output.csv")
    # save a lite version for axon synthesis
    df_out_lite = df_out[
        ["source", "brain_region_id", "hemisphere", "population_id", "probability"]
    ]
    df_out_lite.to_csv(out_path + "clustering_output_AS.csv")

    df_conn = pd.DataFrame(
        connexion_prob,
        columns=[
            "source",
            "source_brain_region_id",
            "source_hemisphere",
            "class_id",
            "source_population_id",
            "target_region",
            "target_brain_region_id",
            "target_population_id",
            "target_hemisphere",
            "probability",
            "feature_means",
        ],
    )
    df_conn.to_csv(out_path + "conn_probs.csv")
    df_post.to_csv(out_path + "posteriors.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config_ = configparser.ConfigParser()
    config_.read(sys.argv[1])

    run_classification(config_, config_["validation"]["verify_classification"] == "True")
