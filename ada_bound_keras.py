from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


class AdaBound(tf.keras.optimizers.Optimizer):
    """
    https://openreview.net/pdf?id=Bkg3g2R9FX
    """

    def __init__(self,
                 lr=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 final_lr=0.1, 
                 epsilon=None,
                 decay=0, 
                 amsbound=False,
               **kwargs):
        super(AdaBound, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.final_lr = K.variable(final_lr, name='final_lr')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
            
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.amsbound = amsbound
        
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        t = math_ops.cast(self.iterations, K.floatx()) + 1
        
        lr_t = self.lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
             (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsbound:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsbound:
                vhat_t = K.maximum(vhat, v_t)
                denom = K.sqrt(vhat_t) + self.epsilon
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                denom = K.sqrt(v_t) + self.epsilon

            
            lower_bound = self.final_lr * (1 - 1 / ((1-self.beta_2) * (t + 1)))
            upper_bound = self.final_lr * (1 + 1 / ((1-self.beta_2) * (t + 1)))
            eta_hat = K.minimum(K.maximum(lr_t/denom, lower_bound), upper_bound)
#             eta = eta_hat / K.sqrt(t)
            
            p_t = p -  m_t * eta_hat
        
            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'final_lr': float(K.get_value(self.final_lr)),
            'epsilon': self.epsilon,
            'amsbound': self.amsbound,
            }
        base_config = super(AdaBound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
