from ModelAndTest import *

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
mnist = input_data.read_data_sets("Fashion_data", one_hot=True)

for i in range(5):
    for lrab in [1.0, 0.1, 0.01, 0.001]:
        tf.reset_default_graph()

        model = Model3_mnist_bn_split()
        model.learning_rate_ab = lrab # 1.0, 0.1, 0.01, 0.001
        taskname = 'fmnist_m3_bnsplr2_ab%g_T%d' % (model.learning_rate_ab, i)
        #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
        tensorboard_dir = '/hpctmp/matcyon/Results-fmnist-m3/%s/' % taskname
        test = Test()
        test.test_lr(model=model, dataset=mnist, 
                     lr_list=np.logspace(-3,3,40), max_step=1200, # re do
                     logdir=tensorboard_dir)
        test.value_check()
        data_save([test.lr_list,test.value_history_np], './fmnist-m3/%s.dat'%taskname)