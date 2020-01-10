# coding: utf-8
"""
This is a generalized version of the atol code written by Martin Royer
@author: Martin Royer and Diego Fiori
@copyright: INRIA 2019
"""

import numpy as np

from sklearn.metrics import pairwise
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.cluster import KMeans

from joblib import Parallel, delayed

import warnings

from sklearn.utils.validation import check_is_fitted


def _compute_inertias(cluster_model, arrays):
    cluster_center = cluster_model(n_clusters=1).fit(arrays).cluster_centers_
    # sqrt removed since not present in the original paper
    return np.sum(np.linalg.norm(arrays.reshape(-1, 2) - cluster_center.reshape(-1, 2), ord=2, axis=1)**2)


def centers_and_inertias(diags, n_centers, cluster_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clusters = cluster_model(n_clusters=n_centers).fit(diags)
        if np.size(np.unique(clusters.labels_)) < n_centers:
            clusters = cluster_model(n_clusters=np.size(np.unique(clusters.labels_))).fit(diags)
    inertias = np.array([_compute_inertias(cluster_model, diags[clusters.labels_ == lab, :])
                         for lab in np.unique(clusters.labels_)])
    return clusters, inertias


def lapl_feats(diags, centers, inertias, eps=1e-10):
    return np.exp(-np.sqrt(pairwise.pairwise_distances(diags, Y=centers) / (inertias + eps)))


def gaus_feats(diags, centers, inertias, eps=1e-10):
    return np.exp(-pairwise.pairwise_distances(diags, Y=centers) / (inertias + eps))


class Atol(BaseEstimator, ClusterMixin, TransformerMixin):
    """Atol learning
        Read more in [https://arxiv.org/abs/1909.13472]
    """

    def __init__(self, n_centers=5, cluster_model=None, method=lapl_feats, aggreg=np.sum, order=None,
                 ignore_hom_dim=False, padding=None, n_jobs=None):
        self.n_centers = n_centers
        self.cluster_model = cluster_model if cluster_model is not None else KMeans
        self.method = method
        self.aggreg = aggreg
        self.order = order
        self.ignore_hom_dim = ignore_hom_dim
        self.padding = padding
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        self.centers_ = []
        self.inertias_ = []
        if self.padding is not None:
            padding = np.array([self.padding, self.padding])
            X = np.array([x[np.any(x[:, :-1] != padding, axis=1)] for x in X])
        diags = np.concatenate([diag for diag in X])
        self.hom_dims_ = np.unique(diags[:, -1]).astype(int)
        len_hom = len(self.hom_dims_)
        hom_dims = self.hom_dims_
        if self.ignore_hom_dim:
            hom_dims = np.zeros((len_hom, )).astype(int)
            diags[:, -1] = np.zeros(diags[:, -1].shape).astype(int)
        for hom_dim in hom_dims:
            sub_diags = diags[diags[:, -1] == hom_dim][:, :-1]
            clusters, inertias = centers_and_inertias(diags=sub_diags, n_centers=self.n_centers,
                                                      cluster_model=self.cluster_model)
            self.centers_.append(clusters.cluster_centers_)
            self.inertias_.append(inertias)
            if self.ignore_hom_dim:
                self.centers_ = self.centers_*len_hom
                self.inertias_ = self.inertias_*len_hom
                break
        self._is_fitted = True
        return self

    def transform_single_diag(self, diag):
        if self.padding is not None:
            padding = np.array([self.padding, self.padding])
            diag = diag[np.any(diag[:, :-1] != padding, axis=1)]
        diag_atol = [self.aggreg(self.method(diag[diag[:, -1] == hom_dim][:, :-1],
                                             self.centers_[hom_dim],
                                             self.inertias_[hom_dim]), axis=0)
                     if np.sum(diag[:, -1] == hom_dim) > 0
                     else np.zeros(len(self.centers_[hom_dim]))
                     for hom_dim in self.hom_dims_]
        if self.order is not None:
            try:
                diag_atol = np.linalg.norm(diag_atol, ord=self.order, axis=0)
            except ValueError:
                for i, el in enumerate(diag_atol):
                    new_el = np.zeros(self.n_centers)
                    new_el[:len(el)] = el
                    diag_atol[i] = new_el
                diag_atol = np.linalg.norm(diag_atol, ord=self.order, axis=0)
        else:
            diag_atol = np.concatenate(diag_atol)
        return diag_atol

    def transform(self, X, y=None):
        check_is_fitted(self, ['_is_fitted'])
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        Xt = Parallel(n_jobs=n_jobs)(delayed(self.transform_single_diag)(X[i]) for i in range(len(X)))
        return np.array(Xt)
