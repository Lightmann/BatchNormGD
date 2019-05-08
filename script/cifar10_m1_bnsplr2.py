from ModelAndTest import *
from Cifar10 import * 

dataset = Cifar10(dirpath='./cifar10_data', one_hot=True, normalize=False)

for i in range(5):
    for lrab in [1.0, 0.1, 0.01, 0.001]:
    
        tf.reset_default_graph()

        model = Model_cifar10_bn_split()
        model.learning_rate_ab = lrab # 1.0, 0.1, 0.01, 0.001
        taskname = 'cifar10_m1_bnsplr2_ab%g_T%d' % (model.learning_rate_ab, i)
        #tensorboard_dir = '/home/lightmann/Results/%s/' % taskname
        tensorboard_dir = '/hpctmp/matcyon/Results-cifar10-m1/%s/' % taskname
        test = Test()
        test.test_lr(model=model, dataset=dataset, 
                     lr_list=np.logspace(-3,3,40), max_step=1000,
                     logdir=tensorboard_dir)
        test.value_check()
        data_save([test.lr_list,test.value_history_np], './cifar10-m1/%s.dat'%taskname)