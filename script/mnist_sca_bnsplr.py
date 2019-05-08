from ModelAndTest import *

from tensorflow.examples.tutorials.mnist import input_data
dataset = input_data.read_data_sets("MNIST_data", one_hot=True)

for i in range(5):
    for lrab in [1.0]: # , 0.1, 0.01, 0.001
        for aug in np.logspace(-2,2,9):#  [0.01,0.1,1,10]:
            tf.reset_default_graph()

            model = Model_mnist_bn_split()
            model.learning_rate_ab = lrab
            model.set_scaling(weight_aug=aug)
            taskname = 'mnist_m1_sca_bnsplr_add3_ab%g_aug%g_T%d' % (lrab,aug,i)
            #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
            tensorboard_dir = '/hpctmp/matcyon/Results-mnist-m1/%s/' % taskname
            test = Test()
            test.test_lr(model=model, dataset=dataset, 
                         lr_list=np.logspace(-4,4,20)*aug**2, max_step=600,
                         logdir=tensorboard_dir)
            test.value_check()
            data_save([test.lr_list,test.value_history_np], './mnist-m1/%s.dat'%taskname)         