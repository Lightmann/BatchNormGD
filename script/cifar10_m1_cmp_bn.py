from ModelAndTest import *
from Cifar10 import * 

dataset = Cifar10(dirpath='./cifar10_data', one_hot=True, normalize=False)

lr_list = [0.001,0.01,0.1,1,10]
for i in range(5):
    tf.reset_default_graph()

    model = Model_cifar10_bn()
    taskname = 'cifar10_m1_cmp_bn_T%d' % i
    #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
    tensorboard_dir = '/hpctmp/matcyon/Results-cifar10-m1/%s/' % taskname
    test = Test()
    test.test_lr(model=model, dataset=dataset, 
                 lr_list=lr_list, max_step=10000, # max_step
                 logdir=tensorboard_dir)
    test.value_check()
    data_save([test.lr_list,test.value_history_np], './cifar10-m1/%s.dat'%taskname)