from typing import List, Callable, Union
import tensorflow as tf

from .model import Model

from .utils import mixed_dropout
from .propagation import PPRExact, PPRPowerIteration

sparse_dot = tf.sparse_tensor_dense_matmul


class PPNP(Model):
    def _build_layer(
            self, X: Union[tf.Tensor, tf.SparseTensor], out_size: int,
            activation: Callable[[tf.Tensor, str], tf.Tensor],
            regularize: bool = True, keep_prob: float = 0.5) -> tf.Tensor:
        W = tf.get_variable(
                'weights',
                [X.get_shape()[1], out_size],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())
        if regularize:
            self.reg_vars.append(W)

        # drop the input elements for generalisation
        X_drop = mixed_dropout(X, keep_prob)
        

        # X multiplied with weights
        # as its matrix multiplication, we need R1*S x S*C2
        # hence we reshape the weights tensor to S*? in tf.get_variable above, where ? is the number of features that we need per node

        # inputs can be a sparse tensor due to dropout
        # weights will always be there
        
        if isinstance(X, tf.SparseTensor):
            Z_inner = sparse_dot(X_drop, W)
        else:
            # @ is for matrix multiplication!
            Z_inner = X_drop @ W
        
        return activation(Z_inner)

    def build_model(
            self, propagation: Union[PPRExact, PPRPowerIteration],
            hiddenunits: List[int] = [16], reg_lambda: float = 1e-3,
            learning_rate: float = 0.01, keep_prob: float = 0.5,
            activation_fn: Callable[[tf.Tensor, str], tf.Tensor] = tf.nn.relu):
        self.isTrain = tf.placeholder(tf.bool, [], name='isTrain')
        self.idx = tf.placeholder(tf.int32, [None], name='idx')
        self.propagation = propagation
        self.hiddenunits = hiddenunits

        keep_prob = tf.maximum(tf.cast(~self.isTrain, tf.float32), keep_prob)

        # attribute matrix over the layers
        # this holds the node attributes for each hidden layer
        # [0] is initial attributes
        self.Zs = [self.attr_mat_norm]

        # Hidden layers
        for i, hiddenunit in enumerate(self.hiddenunits):

            with tf.variable_scope(f'layer_{i}'):
                first_layer = i == 0
                keep_prob_current = keep_prob if first_layer else 1.
                # append the result of 'pr
                self.Zs.append(
                    self._build_layer(
                        self.Zs[-1], hiddenunit,
                        activation=activation_fn,
                        # only regularise first layer
                        # TODO why?
                        regularize=first_layer,
                        keep_prob=keep_prob_current))

        # Last layer
        # classification layer
        # this is the black box that we have built for ourselves.

        # its just using a perceptron
        with tf.variable_scope(f'layer_{len(self.hiddenunits)}'):
            self.logits_local = self._build_layer(
                    self.Zs[-1], self.nclasses,
                    activation=lambda x: x,
                    regularize=False, keep_prob=keep_prob)

        # Propagation
        # we only propagate the labels
        
        self.logits = self.propagation.build_model(self.logits_local, keep_prob)

        self._build_loss(reg_lambda)
        self._build_training(learning_rate)
        self._build_results()
