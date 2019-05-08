from ModelAndTest import *
from Cifar10 import * 

dataset = Cifar10(dirpath='./cifar10_data', one_hot=True, normalize=True)

for i in range(5):
    tf.reset_default_graph()

    model = Model_cifar10_gd()
    taskname = 'cifar10_m1_gdlr2_T%d' % i
    #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
    tensorboard_dir = '/hpctmp/matcyon/Results-cifar10-m1/%s/' % taskname
    test = Test()
    test.test_lr(model=model, dataset=dataset, 
                 lr_list=np.logspace(-3,1,20), max_step=1000,
                 logdir=tensorboard_dir)
    test.value_check()
    data_save([test.lr_list,test.value_history_np], './cifar10-m1/%s.dat'%taskname)