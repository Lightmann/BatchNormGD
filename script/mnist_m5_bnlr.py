
from ModelAndTest import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

dataset = mnist
for i in range(5):
    for hidden_size in [10,100,1000]:
        tf.reset_default_graph()
        
        model = Model5_mnist_bn(hidden_size=hidden_size)
        taskname = 'mnist_m5_h%d_bnlr_T%d' % (hidden_size,i)
        #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
        tensorboard_dir = '/hpctmp/matcyon/Results-mnist-m5/%s/' % taskname
        test = Test()
        test.test_lr(model=model, dataset=dataset, 
                     lr_list=np.logspace(-3,6,40), max_step=6000,
                     #lr_list=np.logspace(-3,5,40), max_step=6000,
                     #lr_list=np.logspace(-3,1,20), max_step=6000, #'mnist_m4_gdlr_T%d' % i
                     logdir=tensorboard_dir)
        test.value_check()
        data_save([test.lr_list,test.value_history_np], './mnist-m5/%s.dat'%taskname)
print('Over')