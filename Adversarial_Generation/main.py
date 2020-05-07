import tensorflow as tf
import numpy as np
import pickle
from cleverhans.future.tf2.attacks import fast_gradient_method
from network_model import neural_net
import os 

from tensorflow.keras.datasets import mnist


# Loading MNIST dataset
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.reshape(tf.Variable(x_train, dtype=tf.float32), [len(x_train), 784])
x_test = tf.reshape(tf.Variable(x_test, dtype=tf.float32), [len(x_test), 784])
x_train = x_train / 255
x_test = x_test / 255

# Loading Neural Network
obj = neural_net(
        weight_file="Weights/weight_dict",
        data_path="../data/fashion",
        seed=23,
        n_hidden=512,
        patience=36,
        learning_rate=0.001,
    )
obj.display_step = 10
obj.load_data()

# Training Neural Network
# obj.net_initialize()
# obj.train_network()

# Saving trained weights
# print("Saving Trained Weights")
# obj.save_weight()

obj.load_weight()

# FUNCTION : fast_gradient_method(model_fn, x, eps, norm, clip_min=None, clip_max=None, y=None,
#                          targeted=False, sanity_checks=False):
# 
# PARAMETERS :
#   Tensorflow 2.0 implementation of the Fast Gradient Method.
#   :param model_fn: a callable that takes an input tensor and returns the model logits.
#   :param x: input tensor.
#   :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
#   :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
#   :param clip_min: (optional) float. Minimum float value for adversarial example components.
#   :param clip_max: (optional) float. Maximum float value for adversarial example components.
#   :param y: (optional) Tensor with true labels. If targeted is true, then provide the
#             target label. Otherwise, only provide this parameter if you'd like to use true
#             labels when crafting adversarial samples. Otherwise, model predictions are used
#             as labels to avoid the "label leaking" effect (explained in this paper:
#             https://arxiv.org/abs/1611.01236). Default is None.
#   :param targeted: (optional) bool. Is the attack targeted or untargeted?
#             Untargeted, the default, will try to make the label incorrect.
#             Targeted will instead try to move in the direction of being more like y.
#   :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
#             memory or for unit tests that intentionally pass strange input)
limit = 20000
adv_input = x_train[:limit]
eps = 0.1
Z = fast_gradient_method(obj.network,adv_input,eps,np.inf,clip_min=0.,clip_max=1.)
print("####################################\nAdversarial example : \n",Z)

# Saving Adversarial Examples
file = open("adv_data.pkl",'wb')
pickle.dump(Z,file)
file.close()

# Printing Accuracy
pred = obj.network(Z)
accu = obj.accuracy(pred,y_train[:limit])
pred2 = obj.network(x_test)
accu2 = obj.accuracy(pred2,y_test)
print(f"Original data  accuracy : {accu2:<.3f}")
print(f"Adversial example accuracy : {accu:<.3f}")

