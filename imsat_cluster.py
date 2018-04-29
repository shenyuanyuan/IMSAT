import argparse, sys
import numpy as np
import tensorflow as tf
import os
import metrics
os.environ['CUDA_VISIBLE_DEVICES']="1"

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='which gpu device to use', default=1)
parser.add_argument('--lam', type=float, help='trade-off parameter for mutual information and smooth regularization',
                    default=0.2)
parser.add_argument('--mu', type=float, help='trade-off parameter for entropy minimization and entropy maximization',
                    default=4)
parser.add_argument('--batch_size', type=int, help='batch size', default=64)
parser.add_argument('--dataset', type=str, help='which dataset to use', default='mnist')
parser.add_argument('--hidden_dim', type=int, help='hidden size list', default='1200')
args = parser.parse_args()

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
    return new_variables

def fc_layer(input_layer, output_dim):
    input_dim = input_layer.get_shape()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(stddev = 0.01))
    fc_b = create_variables(name='fc_bias', shape=[output_dim], initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def batch_normalization_layer(input_layer, dimension, phase_train, decay = 0.90):
    beta = create_variables(name = 'beta', shape=[dimension], initializer=tf.zeros_initializer())
    gamma = create_variables(name = 'gamma', shape=[dimension], initializer=tf.ones_initializer())
    ema_mean = create_variables(name='ema_mean', shape=[dimension], initializer=tf.ones_initializer())
    ema_var = create_variables(name='ema_var', shape=[dimension], initializer=tf.ones_initializer())
    batch_mean, batch_var = tf.nn.moments(input_layer, [0])
    def mean_var_with_update():
        ema_mean_op = tf.assign(ema_mean, ema_mean * decay + batch_mean * (1 - decay))
        ema_var_op = tf.assign(ema_var, ema_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([ema_mean_op, ema_var_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
    return tf.nn.batch_normalization(input_layer, mean, var, beta, gamma, 2e-5)

def encoder(input, phase_train, n_class = 10, reuse = False):
    with tf.variable_scope('layer1', reuse=reuse):
        l1 = fc_layer(input, hidden_dim)
        relu1 = tf.nn.relu(l1)
        bn1 = batch_normalization_layer(relu1, hidden_dim, phase_train=phase_train)
    with tf.variable_scope('layer2', reuse=reuse):
        l2 = fc_layer(bn1, hidden_dim)
        relu2 = tf.nn.relu(l2)
        bn2 = batch_normalization_layer(relu2, hidden_dim, phase_train=phase_train)
    with tf.variable_scope('layer3', reuse=reuse):
        l3 = fc_layer(bn2, n_class)
    return l3

def entropy(p):
    return -tf.reduce_sum(p * tf.log(p + 1e-16), axis=1)

def compute_kld(p_logit, q_logit):
    p = tf.nn.softmax(p_logit)
    q = tf.nn.softmax(q_logit)
    return tf.reduce_sum(p*(tf.log(p + 1e-16) - tf.log(q + 1e-16)), axis=1)

def generate_virtual_adversarial_perturbation(x, ul_logits, phase_train, xi=10, Ip=1):
    d = tf.random_normal(shape=tf.shape(x))
    d /= (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0), axis=1)), [-1, 1]) + 1e-16)
    for ip in range(Ip):
        y1 = ul_logits
        y2 = encoder(x + xi*d, phase_train)
        kl_loss = tf.reduce_mean(compute_kld(y1, y2))
        grad = tf.gradients(kl_loss, [d])[0]
        d = tf.stop_gradient(grad)
        d /= (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0), axis=1)), [-1, 1]) + 1e-16)
    return d

def virtual_adversarial_loss(x, ul_logits, phase_train):
    r_vadv = generate_virtual_adversarial_perturbation(x, ul_logits, phase_train)
    ul_logits = tf.stop_gradient(ul_logits)
    y1 = ul_logits
    y2 = encoder(x + r_vadv, phase_train)
    return tf.reduce_mean(compute_kld(y1, y2))

def build_train_graph(ul_x, phase_train, learning_rate):
    p_logit = encoder(ul_x, phase_train)
    p = tf.nn.softmax(p_logit)
    p_ave = tf.reduce_mean(p, axis=0)
    loss_eq2 = -tf.reduce_sum(p_ave * tf.log(p_ave + 1e-16))
    loss_eq1 = tf.reduce_mean(entropy(p))
    loss_eq = loss_eq1 - args.mu * loss_eq2
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        ul_logits = encoder(ul_x, phase_train)
        loss_vat = virtual_adversarial_loss(ul_x, ul_logits, phase_train)
    total_loss = loss_vat + args.lam * loss_eq
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    tvars = tf.trainable_variables()
    gvs = optimizer.compute_gradients(total_loss, tvars)
    capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    return total_loss, loss_eq1, loss_vat, train_op

def build_test_graph(ul_x, phase_train):
    prob = tf.nn.softmax(encoder(ul_x, phase_train, reuse=True))
    pred = tf.argmax(prob, axis=1)
    return pred

if args.dataset == 'mnist':
    sys.path.append('mnist')
    from load_mnist import *
    dataset = load_mnist_whole(PATH='mnist/', scale=1.0 / 255.0)
else:
    print('The dataset is not supported.')
    raise NotImplementedError

train_num = len(dataset.images_train)
test_num = len(dataset.images_test)
total_iterations_train = train_num // args.batch_size
input_dim = dataset.images_train.shape[1]
output_dim = np.max(dataset.labels_train) + 1
hidden_dim = args.hidden_dim
batchsize_ul = 64
test_batchsize_ul = 100
n_epoch = 500

def main():
    unlabeled_x = tf.placeholder(tf.float32, shape=[None, input_dim])
    is_training = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)
    loss, loss1, loss2, train_op = build_train_graph(unlabeled_x, is_training, learning_rate)
    pred = build_test_graph(unlabeled_x, is_training)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            sum_loss = 0
            sum_conditional_entropy = 0
            sum_rsat = 0

            if epoch == 0:
                lr = 0.01
            else:
                if lr <= 1e-5:
                    lr = 1e-5
                else:
                    lr = lr*0.98
            for it in range(total_iterations_train):
                x_u, _ = dataset.sample_minibatch(batchsize_ul)
                _, ploss, ploss1, ploss2 = sess.run([train_op, loss, loss1, loss2], feed_dict={unlabeled_x: x_u, is_training: True, learning_rate: lr})
                sum_loss = sum_loss + ploss
                sum_conditional_entropy = sum_conditional_entropy + ploss1
                sum_rsat = sum_rsat + ploss2

            print('total loss: %7.4f, conditional entropy: %7.4f, vat loss: %7.4f' %
                  (sum_loss/total_iterations_train, sum_conditional_entropy/total_iterations_train, sum_rsat/total_iterations_train))

            #obtain data
            test_train_data = dataset.images_train
            test_train_labels = dataset.labels_train
            test_train_epoch_num = train_num // test_batchsize_ul
            off_index = 0
            pre_y = np.zeros(train_num)
            true_y = np.zeros(train_num)
            for it in range(test_train_epoch_num):
                if it == test_train_epoch_num:
                    samples = test_train_data[off_index:]
                    labels = test_train_labels[off_index:]
                else:
                    samples = test_train_data[off_index:off_index+test_batchsize_ul]
                    labels = test_train_labels[off_index:off_index+test_batchsize_ul]
                pre_y[off_index:off_index+test_batchsize_ul] = sess.run(pred, feed_dict={unlabeled_x: samples, is_training:False})
                true_y[off_index:off_index+test_batchsize_ul] = labels
                off_index = off_index + test_batchsize_ul
            pre_y = pre_y.astype(np.int32)
            true_y = true_y.astype(np.int32)
            print('test result for epoch (training data): %d' % epoch)
            print(metrics.acc(true_y, pre_y))

            test_test_data = dataset.images_test
            test_test_labels = dataset.labels_test
            test_test_epoch_num = test_num // test_batchsize_ul
            off_index = 0
            pre_y = np.zeros(test_num)
            true_y = np.zeros(test_num)
            for it in range(test_test_epoch_num):
                if it == test_test_epoch_num:
                    samples = test_test_data[off_index:]
                    labels = test_test_labels[off_index:]
                else:
                    samples = test_test_data[off_index:off_index + test_batchsize_ul]
                    labels = test_test_labels[off_index:off_index + test_batchsize_ul]
                pre_y[off_index:off_index + test_batchsize_ul] = sess.run(pred, feed_dict={unlabeled_x: samples,
                                                                                           is_training: False})
                true_y[off_index:off_index + test_batchsize_ul] = labels
                off_index = off_index + test_batchsize_ul
            pre_y = pre_y.astype(np.int32)
            true_y = true_y.astype(np.int32)
            print('test result for epoch (test data): %d' % epoch)
            print(metrics.acc(true_y, pre_y))

if __name__ == '__main__':
    main()



