import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from .utils import sparse_matrix_to_tensor, sparse_dropout

sparse_dot = tf.sparse_tensor_dense_matmul

# The normalised laplacian matrix
# https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf
def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    # calculate the sum along columns
    # this means that we reduce the NxN matrix into Nx1
    # .A1 converts the reduced matrix (nnodes x 1) to a vector
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    # M is the normalised Laplacian matrix
    A_inner = sp.eye(nnodes) - (1 - alpha) * M

    return alpha * np.linalg.inv(A_inner.toarray())


class PPRExact:
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float):
        self.alpha = alpha
        self.ppr_mat = calc_ppr_exact(adj_matrix, alpha)

    def build_model(self, Z: tf.Tensor, keep_prob: float) -> tf.Tensor:
        with tf.variable_scope(f'Propagation'):
            
            # is this the matrix with which we have to multiply 
            # for propagating our labels to the neighbours?
            
            ppr_mat_tf = tf.constant(self.ppr_mat, dtype=tf.float32)

            # this random dropping in the ppr can be the random features
            # that makes our neural network stronger

            ppr_drop = tf.nn.dropout(ppr_mat_tf, keep_prob)

            # check how we get a transition matrix to compute pagerank's
            # next iteration
            return ppr_drop @ Z


class PPRPowerIteration:
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int):
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.A_hat = (1 - alpha) * M

    def build_model(self, Z: tf.Tensor, keep_prob: float) -> tf.Tensor:
        with tf.variable_scope(f'Propagation'):
            A_hat_tf = sparse_matrix_to_tensor(self.A_hat)
            Zs_prop = Z
            for _ in range(self.niter):
                A_drop = sparse_dropout(A_hat_tf, keep_prob)

                # wtf, why do we have self.alpha*Z inside the loop?
                Zs_prop = sparse_dot(A_drop, Zs_prop) + self.alpha * Z
            return Zs_prop

