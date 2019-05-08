from ModelAndTest import *

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
mnist = input_data.read_data_sets("Fashion_data", one_hot=True)

for i in range(5):
    tf.reset_default_graph()

    model = Model_mnist_gd()
    taskname = 'fmnist_m1_gdlr_T%d' % i
    #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
    tensorboard_dir = '/hpctmp/matcyon/Results-fmnist-m1/%s/' % taskname
    test = Test()
    test.test_lr(model=model, dataset=mnist, 
                 lr_list=np.logspace(-3,1,20), max_step=1200,
                 logdir=tensorboard_dir)
    test.value_check()
    data_save([test.lr_list,test.value_history_np], './fmnist-m1/%s.dat'%taskname)