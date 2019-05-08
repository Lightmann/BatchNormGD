
from ModelAndTest import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

dataset = mnist
for i in range(5):
    for lrab in [10,1,0.1]:
        for hidden_size in [10,100,1000]:
            tf.reset_default_graph()
        
            model = Model5_mnist_bn_split(hidden_size=hidden_size)
            model.learning_rate_ab = lrab
            taskname = 'mnist_m5_h%d_bnsplr_ab%g_T%d' % (hidden_size,lrab,i)
            #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
            tensorboard_dir = '/hpctmp/matcyon/Results-mnist-m5/%s/' % taskname
        
            test = Test()
            test.test_lr(model=model, dataset=dataset, 
                         #lr_list=np.logspace(-3,8,50), max_step=6000,
                         lr_list=np.logspace(-3,7,40), max_step=3000,
                         #lr_list=np.logspace(-3,7,40), max_step=6000,
                         #lr_list=np.logspace(-3,5,40), max_step=6000,
                         #lr_list=np.logspace(-3,1,20), max_step=6000, #'mnist_m4_gdlr_T%d' % i
                         logdir=tensorboard_dir)
            test.value_check()
            data_save([test.lr_list,test.value_history_np], './mnist-m5/%s.dat'%taskname)
print('Over')