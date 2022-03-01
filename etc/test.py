import tensorflow as tf
import timeit

def cpu_gpu_compare(n):
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([10,n])
        cpu_b = tf.random.normal([n,10])
    print(cpu_a.device,cpu_b.device)
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([10,n])
        gpu_b = tf.random.normal([n,10])
    print(gpu_a.device,gpu_b.device)
    def cpu_run():
        with tf.device('/cpu:0'):              
            c = tf.matmul(cpu_a,cpu_b)
        return c
    def gpu_run():
        with tf.device('/gpu:0'):           
            c = tf.matmul(gpu_a,gpu_b)
        return c

    cpu_time = timeit.timeit(cpu_run,number=10)
    gpu_time = timeit.timeit(gpu_run,number=10)
    print('warmup:',cpu_time,gpu_time)

    cpu_time = timeit.timeit(cpu_run,number=10)
    gpu_time = timeit.timeit(gpu_run,number=10)
    print('run_time:',cpu_time,gpu_time)
    return cpu_time,gpu_time
n_list2 = range(2001,1000000,1000)
n_list = list(n_list2)
time_cpu =[]
time_gpu =[]
for n in n_list:
    t=cpu_gpu_compare(n)
    time_cpu.append(t[0])
    time_gpu.append(t[1])

for line in list(zip(time_cpu, time_gpu)):
    print(line)