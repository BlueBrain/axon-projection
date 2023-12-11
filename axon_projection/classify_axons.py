import logging
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import sys

def gmm_bic_score(estimator, data):
    """ Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to *maximize*
    return -estimator.bic(data)

def find_best_gmm(data, n_components_max, seed=None, n_jobs=12):
    """ Takes in input the data on which we want to build a Gaussian Mixture Model (GMM), and the maximum number of mixture components desired. Returns the best number of components and covariance type to use, by minimizing the BIC score on the data.

    Args:
        data (list numpy array): the feature vectors on which to base the GMM. In this case, we are using number of terminal points for each target region.
        n_components_max (int) : maximum number of mixtures to use (i.e. number of Gaussians). The algorithm will use at most n_components_max classes for the clustering.
        seed (int) : random seed, for reproducibility. Seed is used for the random initialization of the GMMs (as is done in K-means algorithm)
        n_jobs (int) : number of processes used in the search.

    Returns:
        grid_search.best_params (dict) : parameters of the best GMM.
        res (pandas dataframe) : output of the search."""
    
    param_grid = {
        "n_components": range(1, n_components_max+1),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(random_state=seed), param_grid=param_grid, scoring=gmm_bic_score, verbose=0, n_jobs=n_jobs
    )
    grid_search.fit(data)
    res = pd.DataFrame(grid_search.cv_results_)[["param_n_components", "param_covariance_type", "mean_test_score"]]
    res["mean_test_score"] = -res["mean_test_score"]
    res = res.rename(
        columns={
            "param_n_components": "n_components",
            "param_covariance_type": "covariance_type",
            "mean_test_score": "BIC",
        }
    )

    return grid_search.best_params_, res

def run_classification(config):
    """ Creates the projectional classes for each source region. 
    
    Inputs:
        Needs a file 'axonal_projections.csv' that contains the feature data for classification.
    
    Outputs:
        clustering_output.csv: the GMM parameters for each class (probability to belong to class, means, variances), with number of points in class.
        conn_probs.csv: the probability of connection to target regions for each class. This probability is computed as the mean number of terminal for that target region and class, divided by the total number of terminal points for that class.
    """
    data_path = config["output"]["path"]+ "axonal_projections_"+config["morphologies"]["hierarchy_level"]+".csv"
    # load data containing source region ("source") and number of terminals in each target region ("<region_acronym>")
    f = pd.read_csv(data_path)
    # first two columns are row_id and source regions, the rest is target regions
    target_regions_names = f.columns[2:]
    # for reproducibility
    seed = int(config["random"]["seed"])
    # this might not be needed at the moment
    np.random.seed = seed
    sources = f.source.unique()   # array of all source regions found in data
    clustering_output = []
    connexion_prob = []
    n_jobs = int(config["classify"]["n_jobs"])
    # do the clustering for each source
    for s_a in sources:
        logging.info("Source region : %s", s_a)
        # dataframe with source s_a
        f_as = f[f["source"]==s_a]
        
        logging.info("Shape of data %s", f_as.shape)
        n_rows, n_cols = f_as.shape
        # skip for now source regions that have less than five data points (because the data is split in five during the grid_search for the best number of mixtures)
        if n_rows < 5:
            logging.info("Skip source region %s with %s point(s).", s_a, n_rows)
            continue
        # exclude row id and source region in data
        data = np.array(f_as.iloc[:, 2:].values.tolist())

        # we allow at most to have n_max_components classes
        n_max_components = int(n_rows/2.)

        best_params, grid_search_res = find_best_gmm(data, n_max_components, seed=seed, n_jobs=n_jobs)
        # grid_search_res.to_markdown("grid_search_"+s_a+".md")
        logging.info(best_params)
        n_components=best_params['n_components']
        cov_type=best_params['covariance_type']

        # we take at most n_components components for the mixture, but since it is a BayesianGM, it should use the optimal number of components <= n_components
        # gmm = BayesianGaussianMixture(n_components=n_rows, verbose=1, random_state=np.random.seed).fit(data)        

        gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, verbose=1, random_state=seed).fit(data)

        logging.info("GMM params: %s", gmm.get_params())
        for c in range(n_components):
            clustering_output.append([s_a, c, int(n_rows*gmm.weights_[c]), gmm.weights_[c], gmm.means_[c], gmm.covariances_[c]])
            for t in range(len(gmm.means_[c])):
                # don't show probability if it's 0
                if gmm.means_[c][t] < 1e-16:
                    continue
                connexion_prob.append([s_a, c, target_regions_names[t], gmm.means_[c][t]/np.sum(gmm.means_[c])])

        # sns.catplot(
        # data=grid_search_res,
        # kind="bar",
        # x="N_components",
        # y="BIC",
        # hue="Cov_type",
        # )
        # plt.savefig("BIC_scores_"+s_a+".pdf")

    df_out = pd.DataFrame(clustering_output, columns=["source", "class_id", "num_data_points", "probabilities", "means", "variances"])
    df_out.to_csv(config["output"]["path"]+"clustering_output.csv")
    df_conn = pd.DataFrame(connexion_prob, columns=["source", "class_id", "target_region", "probability"])
    df_conn.to_csv(config["output"]["path"]+"conn_probs.csv")

if __name__=="__main__":
    
    logging.basicConfig(level=logging.DEBUG)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    run_classification(config)
    

