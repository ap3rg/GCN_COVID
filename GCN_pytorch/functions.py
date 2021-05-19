import scipy 
import numpy as np
import networkx as nx


def calculate_degree_matrix(A):
    '''
    Takes in adjacency matrix and returns degree matrix. D_ii = SUM_j(A_ij)
    parameters
        - A: Adjacency matrix (SciPy sparse matrix)
    
    returns
        - D: Degree matrix (SciPy sparse matrix)

    '''
    A_triu = scipy.sparse.triu(A)
    degree = A_triu.sum(axis=0)
    D = scipy.sparse.eye(A.shape[0]).multiply(degree)

    return D

def calculate_spectral_norm_matrix(G, nodelist=None):
    # Build adjacency matrix with self-connections A' (A_)
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    A_ = A + scipy.sparse.eye(A.shape[0])

    # Calculate degree matrix of A_
    D_ = calculate_degree_matrix(A_)

    A_norm = D_.power(-0.5).dot(A_).dot(D_.power(-0.5))

    return A_norm