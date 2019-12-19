import numpy as np
from numpy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import gudhi as gd


def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def _get_heat_kernel_filtered_simplex(graph, time=0):
    D_05 = np.diag(graph.sum(axis=1)**-0.5)
    laplacian = np.identity(len(graph)) - np.linalg.multi_dot([D_05, graph, D_05])
    eig_values, eig_vectors = eigh(laplacian)
    signature = hks_signature(eig_vectors, eig_values, time)
    return signature


def _get_base_simplex(A):
    num_vertices = A.shape[0]
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
        for j in range(i + 1, num_vertices):
            if A[i, j] > 0:
                st.insert([i, j], filtration=-1e10)
    return st.get_filtration()


def apply_graph_extended_persistence(A, filtration_val):
    """ The method is written by the author of the perslay paper and can (and should) be improved!
    Furthermore it needs gudhi in order to be run! We need a giotto implementation of the simplex tree!!!
    """
    basesimplex = _get_base_simplex(A)
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    num_edges = len(xs)

    if len(filtration_val.shape) == 1:
        min_val, max_val = filtration_val.min(), filtration_val.max()
    else:
        min_val = min([filtration_val[xs[i], ys[i]] for i in range(num_edges)])
        max_val = max([filtration_val[xs[i], ys[i]] for i in range(num_edges)])

    st = gd.SimplexTree()
    st.set_dimension(2)

    for simplex, filt in basesimplex:
        st.insert(simplex=simplex + [-2], filtration=-3)

    if len(filtration_val.shape) == 1:
        if max_val == min_val:
            fa = -.5 * np.ones(filtration_val.shape)
            fd = .5 * np.ones(filtration_val.shape)
        else:
            fa = -2 + (filtration_val - min_val) / (max_val - min_val)
            fd = 2 - (filtration_val - min_val) / (max_val - min_val)
        for vid in range(num_vertices):
            st.assign_filtration(simplex=[vid], filtration=fa[vid])
            st.assign_filtration(simplex=[vid, -2], filtration=fd[vid])
    else:
        if max_val == min_val:
            fa = -.5 * np.ones(filtration_val.shape)
            fd = .5 * np.ones(filtration_val.shape)
        else:
            fa = -2 + (filtration_val - min_val) / (max_val - min_val)
            fd = 2 - (filtration_val - min_val) / (max_val - min_val)
        for eid in range(num_edges):
            vidx, vidy = xs[eid], ys[eid]
            st.assign_filtration(simplex=[vidx, vidy], filtration=fa[vidx, vidy])
            st.assign_filtration(simplex=[vidx, vidy, -2], filtration=fd[vidx, vidy])
        for vid in range(num_vertices):
            if len(np.where(A[vid, :] > 0)[0]) > 0:
                st.assign_filtration(simplex=[vid], filtration=min(fa[vid, np.where(A[vid, :] > 0)[0]]))
                st.assign_filtration(simplex=[vid, -2], filtration=min(fd[vid, np.where(A[vid, :] > 0)[0]]))

    st.make_filtration_non_decreasing()
    distorted_dgm = st.persistence()
    normal_dgm = dict()
    normal_dgm["Ord0"], normal_dgm["Rel1"], normal_dgm["Ext0"], normal_dgm["Ext1"] = [], [], [], []
    for point in range(len(distorted_dgm)):
        dim, b, d = distorted_dgm[point][0], distorted_dgm[point][1][0], distorted_dgm[point][1][1]
        pt_type = "unknown"
        if (-2 <= b <= -1 and -2 <= d <= -1) or (b == -.5 and d == -.5):
            pt_type = "Ord" + str(dim)
        if (1 <= b <= 2 and 1 <= d <= 2) or (b == .5 and d == .5):
            pt_type = "Rel" + str(dim)
        if (-2 <= b <= -1 and 1 <= d <= 2) or (b == -.5 and d == .5):
            pt_type = "Ext" + str(dim)
        if np.isinf(d):
            continue
        else:
            b, d = min_val + (2 - abs(b)) * (max_val - min_val), min_val + (2 - abs(d)) * (max_val - min_val)
            if b <= d:
                normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([b, d])]))
            else:
                normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([d, b])]))

    dgmOrd0 = np.array([normal_dgm["Ord0"][point][1] for point in range(len(normal_dgm["Ord0"]))])
    dgmExt0 = np.array([normal_dgm["Ext0"][point][1] for point in range(len(normal_dgm["Ext0"]))])
    dgmRel1 = np.array([normal_dgm["Rel1"][point][1] for point in range(len(normal_dgm["Rel1"]))])
    dgmExt1 = np.array([normal_dgm["Ext1"][point][1] for point in range(len(normal_dgm["Ext1"]))])
    if dgmOrd0.shape[0] == 0:
        dgmOrd0 = np.zeros([0, 3])
    else:
        dgmOrd0 = np.concatenate([dgmOrd0, np.zeros((dgmOrd0.shape[0], 1))], axis=1)
    if dgmExt1.shape[0] == 0:
        dgmExt1 = 3*np.ones([0, 3])
    else:
        dgmExt1 = np.concatenate([dgmExt1, 3*np.ones((dgmExt1.shape[0], 1))], axis=1)
    if dgmExt0.shape[0] == 0:
        dgmExt0 = np.ones([0, 3])
    else:
        dgmExt0 = np.concatenate([dgmExt0, np.ones((dgmExt0.shape[0], 1))], axis=1)
    if dgmRel1.shape[0] == 0:
        dgmRel1 = 2*np.ones([0, 3])
    else:
        dgmRel1 = np.concatenate([dgmRel1, 2*np.ones((dgmRel1.shape[0], 1))], axis=1)
    return np.concatenate([dgmOrd0, dgmExt0, dgmRel1, dgmExt1], axis=0)


def _check_diagram_dimensions(list_of_dgm):
    # we analyze the maximum length of diagrams and their point labels (hom_dim)
    label_n = {}
    for dgm in list_of_dgm:
        dgm_labels = np.unique(dgm[:, -1])
        for label in dgm_labels:
            label_len = np.sum(dgm[:, -1] == label)
            if label in label_n.keys() and label_n[label] < label_len:
                label_n[label] = label_len
            elif label not in label_n.keys():
                label_n[label] = label_len
    # now we want to ensure that each diagram has the same number of points (we replicate points for the shorter)
    unique_labels = label_n.keys()
    max_n_points = np.sum([label_n[key] for key in unique_labels])
    diags = []
    for i, dgm in enumerate(list_of_dgm):
        new_diag = np.zeros((1, max_n_points, 3))
        temp = 0
        for label in unique_labels:
            label_points = dgm[:, -1] == label
            if np.sum(label_points) > 0:
                new_diag[0, temp: temp+np.sum(label_points)] = dgm[label_points]
                if np.sum(label_points) < label_n[label]:
                    new_diag[0, temp+np.sum(label_points):temp+label_n[label]] = new_diag[0, temp+np.sum(label_points)-1]
                temp += label_n[label]
        diags.append(new_diag)
    return np.concatenate(diags)


class GeneralFiltrationGraphHomology(BaseEstimator, TransformerMixin):
    def __init__(self, filtration_fun=None, n_jobs=None, **function_parameters):
        self.filtration_fun = _get_heat_kernel_filtered_simplex if filtration_fun is None else filtration_fun
        self.function_parameters = function_parameters
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        self.vertex_filtration_ = Parallel(n_jobs=self.n_jobs)(delayed(self.filtration_fun)
                                                               (X[i], **self.function_parameters)
                                                               for i in range(len(X)))
        self.X_ = X
        return self

    def transform(self, X, y=None):
        if X != self.X_:
            raise ValueError('The passed array must be the same in both fit and transform methods.')
        Xt = Parallel(n_jobs=self.n_jobs)(delayed(apply_graph_extended_persistence)(X[i], self.vertex_filtration_[i])
                                          for i in range(len(X)))
        return _check_diagram_dimensions(Xt)





