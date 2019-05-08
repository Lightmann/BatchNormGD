from ModelAndTest import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

for i in range(5):
    tf.reset_default_graph()

    model = Model_mnist_bn_split()
    model.learning_rate_ab = 0.001 # 1.5, 1.0, 0.5, 0.1, 0.01, 0.001
    taskname = 'mnist_bnsplr_ab%g_T%d' % (model.learning_rate_ab, i)
    #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
    tensorboard_dir = '/hpctmp/matcyon/Results/%s/' % taskname
    test = Test()
    test.test_lr(model=model, dataset=mnist, 
                 lr_list=np.logspace(-3,3,40), max_step=600,
                 logdir=tensorboard_dir)
    test.value_check()
    data_save([test.lr_list,test.value_history_np], '%s.dat'%taskname)