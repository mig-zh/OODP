import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot
import datetime
import os
import tensorflow.contrib.slim as slim 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = 'cifar10/cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')
DDIM= 10
MODE = 'mmdminmax' # Valid options are dcgan, wgan, or wgan-gp
DIM = 64 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
OPT_ITERS = 100
MAP_ITERS = 1 
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
geniter= 10
inceptioniter = 100
lib.print_model_settings(locals().copy())

def avg_pool(x,n,m):
    return tf.nn.avg_pool(x, ksize=[1, 1, n, n], strides=[1, 1, m, m], padding='VALID',data_format='NCHW')
def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, 1, n, n], strides=[1, 1, n, n], padding='VALID',data_format='NCHW')
def Batchnorm(name, axes, inputs, is_train=tf.Variable(True, trainable=False), 
            decay = 0.9, epsilon = 0.00001, act = tf.identity):#NCHW format
    if ((axes == [0,2,3]) or (axes == [0,1,2])):
        if axes==[0,1,2]: #NHW
            inputs = tf.expand_dims(inputs, 1)
            #axes = [0,2,3]
        # Old (working but pretty slow) implementation:
        ##########

        # inputs = tf.transpose(inputs, [0,2,3,1])

        # mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
        # offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
        # scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
        # result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

        # return tf.transpose(result, [0,3,1,2])

        # New (super fast but untested) implementation:
        inputs = tf.transpose(inputs, [0, 2, 3, 1]) #NCHW -> NHWC

    # else (N, D)

    x_shape = inputs.get_shape()
    params_shape = x_shape[-1:]


    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops

    with tf.variable_scope(name) as vs:
        axis = list(range(len(x_shape) - 1))


        ## 2.
        # if tf.__version__ > '0.12.1':
        #     moving_mean_init = tf.zeros_initializer()
        # else:
        #     moving_mean_init = tf.zeros_initializer

        offset = lib.param(name+'.offset', np.zeros(params_shape, dtype='float32'))
        scale = lib.param(name+'.scale', tf.random_normal(params_shape, mean=1.0, stddev =0.002  ))

        moving_mean = lib.param(name+'.moving_mean', np.zeros(params_shape, dtype='float32'), trainable=False)
        moving_variance = lib.param(name+'.moving_variance', np.ones(params_shape, dtype='float32'), trainable=False)


        ## 3.
        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(inputs, axis)
        

        def mean_var_with_update():
            try:    # TF12
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay, zero_debias=False)     # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay, zero_debias=False) # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay)
                # print("TF11 moving")
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)

        m, v = tf.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        result = act( tf.nn.batch_normalization(inputs, m, v, offset, scale, epsilon))
        
        if ((axes == [0,2,3]) or (axes == [0,1,2])):
            result = tf.transpose(result, [0, 3, 1, 2]) #NHWC -> NCHW

        return result 

def makeScaleMatrix(num_gen, num_orig):

    # first 'N' entries have '1/N', next 'M' entries have '-1/M'
    s1 =  tf.constant(1.0 / num_gen, shape = [num_gen, 1])
    s2 = -tf.constant(1.0 / num_orig, shape = [num_orig, 1])

    return tf.concat([s1, s2], 0)

    """
    Calculates cost of the network, which is square root of the mixture of 'K'
    RBF kernels

    x:       Batch from the dataset
    samples: Samples from the uniform distribution
    sigma:   Bandwidth parameters for the 'K' kernels
    """
#def getmmdloss(x, gen_x, batch_size, sigma = [2]):
def getmmdloss(x, gen_x, batch_size, sigma = [1,10,100 ]):
        # x train \
        # gen_x
        # generate images from the provided uniform samples
        # concatenation of the generated images and images from the dataset
        # first 'N' rows are the generated ones, next 'M' are from the data

    X = tf.concat([gen_x, x], 0)

        # dot product between all combinations of rows in 'X'
    XX = tf.matmul(X, tf.transpose(X))

        # dot product of rows with themselves
    X2 = tf.reduce_sum(X * X, 1, keep_dims = True)

        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
    exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)

        # scaling constants for each of the rows in 'X'
    s = makeScaleMatrix(batch_size, batch_size) #This need N = M

        # scaling factors of each of the kernel values, corresponding to the
        # exponent values
    S = tf.matmul(s, tf.transpose(s))

    loss = 0

        # for each bandwidth parameter, compute the MMD value and add them all
    for i in range(len(sigma)):

            # kernel values for each combination of the rows in 'X' 
        kernel_val = tf.exp(1.0 / sigma[i] * exponent)
        loss += tf.reduce_sum(S * kernel_val)

    return tf.sqrt(loss)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples, noise=None,is_train=None,reuse=False):
   # with tf.variable_scope("foo", reuse=reuse):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
	#noise = tf.random_uniform((n_samples, 128),minval=-1.0,maxval=1.0)
    
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = Batchnorm('Generator.BN1', [0], output,is_train)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = Batchnorm('Generator.BN2', [0,2,3], output,is_train)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = Batchnorm('Generator.BN3', [0,2,3], output,is_train)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)
    '''
    	output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    	output = tf.nn.relu(slim.batch_norm(inputs=output,is_training=is_train,scope='1bn'))
    	output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    	output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    	output = tf.transpose(output,[0,2,3,1])
    	output = tf.nn.relu(slim.batch_norm(inputs=output,is_training=is_train,scope='2bn'))
    	output = tf.transpose(output,[0,3,1,2])


   	output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
   	output = tf.transpose(output,[0,2,3,1])
   	output = tf.nn.relu(slim.batch_norm(inputs=output,is_training=is_train,scope='3bn'))
   	output = tf.transpose(output,[0,3,1,2])


   	output = lib.ops.deconv2d.Deconv2D('Generator.4', DIM, 3, 5, output)
   	output = tf.transpose(output,[0,2,3,1])
   	output = tf.nn.relu(slim.batch_norm(inputs=output,is_training=is_train,scope='4bn'))
   	output = tf.transpose(output,[0,3,1,2])

   	output = tf.tanh(output)
    '''

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs,is_train=None):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = Batchnorm('Discriminator.BN2', [0,2,3], output,is_train)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = Batchnorm('Discriminator.BN3', [0,2,3], output,is_train)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    #output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, DDIM, output)

    return output

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
#real_data = tf.cast(real_data_int, tf.float32)/255.
is_train = tf.placeholder(tf.bool)
fake_data = Generator(BATCH_SIZE,is_train=is_train)

if MODE == 'wgan-gp':
    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)
else:
    disc_real = Discriminator(real_data,is_train=is_train)
    disc_fake = Discriminator(fake_data,is_train=is_train)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)
    mmd_cost = getmmdloss(real_data, fake_data, BATCH_SIZE)

elif MODE == 'wgan-gp':
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
    mmd_cost= getmmdloss(real_data, fake_data, BATCH_SIZE)


#elif MODE == 'dcgan':
elif MODE == 'mmdminmax':
    
    #real_data2= tf.reshape(avg_pool(tf.reshape(real_data,[-1,3,32,32]),4,4), [BATCH_SIZE, -1])
    #fake_data2= tf.reshape(avg_pool(tf.reshape(fake_data,[-1,3,32,32]),4,4), [BATCH_SIZE, -1])
    #real_data3= tf.reshape(avg_pool(tf.reshape(real_data,[-1,3,32,32]),4,2), [BATCH_SIZE, -1])
    #fake_data3= tf.reshape(avg_pool(tf.reshape(fake_data,[-1,3,32,32]),4,2), [BATCH_SIZE, -1])
    #real_data4= tf.reshape(avg_pool(tf.reshape(real_data,[-1,3,32,32]),8,4), [BATCH_SIZE, -1])
    #fake_data4= tf.reshape(avg_pool(tf.reshape(fake_data,[-1,3,32,32]),8,4), [BATCH_SIZE, -1])
    mmdopt_cost =   getmmdloss(disc_real, disc_fake, BATCH_SIZE)
    #mmdmap_cost = -getmmdloss(disc_real, disc_fake, BATCH_SIZE)
    mmdmap_cost =  tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    wd  = - tf.reduce_mean(disc_fake)+ tf.reduce_mean(disc_real)

    mmdopt_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(mmdopt_cost, var_list=gen_params)

    mmdmap_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(mmdmap_cost, var_list=disc_params)


    #mmdcost1= getmmdloss(real_data, fake_data, BATCH_SIZE)
    #mmdcost2= getmmdloss(real_data2, fake_data2, BATCH_SIZE)
    #mmdcost3= getmmdloss(real_data3, fake_data3, BATCH_SIZE)
    #mmdcost4= getmmdloss(real_data4, fake_data4, BATCH_SIZE)
    #mmd_cost=mmdcost1+mmdcost2+mmdcost3+mmdcost4
    #mmd_cost=mmdcost1
    #gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
    #disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
    #disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
    #disc_cost /= 2.
    #mmd_train_op = tf.train.AdamOptimizer().minimize(mmd_cost, var_list=lib.params_with_name('Generator'))
    #gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
    #                                                                              var_list=lib.params_with_name('Generator'))
    #disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
    #                                                                               var_list=lib.params_with_name('Discriminator.'))

# For generating samples
#fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples_128 = Generator(128, noise=fixed_noise,is_train=is_train,reuse=True)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples_128,feed_dict={is_train:False})
    samples = ((samples+1.)*(255./2)).astype('int32')
    tmptime = (datetime.datetime.now()-starttime).seconds
    lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), MODE+'_'+'cifarfig/'+'samples_{}.jpg'.format(frame))

# For calculating inception score
samples_100 = Generator(100,is_train=is_train,reuse=True)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100,feed_dict={is_train:False}))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images in train_gen():
            yield images

# Train loop
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()
    starttime = datetime.datetime.now()
    for iteration in xrange(ITERS):
        start_time = time.time()
        # Train generator
        if iteration >0 :
            for i in xrange(OPT_ITERS):
                _data = gen.next()
            #_,xmmd_cost = session.run([mmd_train_op,mmd_cost],feed_dict={real_data_int: _data,is_train:True})
            #_ = session.run(gen_train_op,feed_dict={is_train:True})
            #xmmd_cost = session.run([mmd_cost],feed_dict={real_data_int: _data,is_train:True})
                wdbeforeG=session.run(wd,feed_dict={is_train:True,real_data_int:_data})

                _ = session.run(mmdopt_train,feed_dict={real_data_int: _data,is_train:True})
                xmmdoptcost = session.run(mmdopt_cost,feed_dict={real_data_int: _data,is_train:True})

                wdafterG=session.run(wd,feed_dict={is_train:True,real_data_int:_data})
            lib.plot.plot('train wd before G',  wdbeforeG)
            lib.plot.plot('train wd after G',  wdafterG)
            lib.plot.plot('train mmd opt',  xmmdoptcost)
 
        # Train critic
        #if MODE == 'dcgan':
        
        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        elif MODE == 'mmdminmax':
            disc_iters = MAP_ITERS
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _ = session.run([ mmdmap_train], feed_dict={real_data_int: _data,is_train:True})
            xmmdmapcost = session.run([mmdmap_cost], feed_dict={real_data_int: _data,is_train:True})
            if MODE == 'wgan':
                _ = session.run(clip_disc_weights)
        lib.plot.plot('train mmd map', xmmdmapcost)
        lib.plot.plot('time', time.time() - start_time)
	#if iteration % geniter ==geniter-1:
                #xmmdcost1,xmmdcost2,xmmdcost3,xmmdcost4 = session.run([mmdcost1,mmdcost2,mmdcost3,mmdcost4],feed_dict={real_data_int: _data})
         #       xmmdcost1 = session.run([mmdcost1],feed_dict={real_data_int: _data,is_train:True})
        #	lib.plot.plot('train mmd1 cost',  xmmdcost1)
        	#lib.plot.plot('train mmd2 cost',  xmmdcost2)
        	#lib.plot.plot('train mmd3 cost',  xmmdcost3)
        	#lib.plot.plot('train mmd4 cost',  xmmdcost4)

        # Calculate inception score every 1K iters
        if iteration % inceptioniter == inceptioniter-1:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % geniter  == geniter-1:
            dev_mmd_costs = []
            dev_wdtest = []
            for images in dev_gen():
                _dev_mmd_cost = session.run( mmdopt_cost, feed_dict={real_data_int: images,is_train:False}) 
                dev_mmd_costs.append(_dev_mmd_cost)

                #_dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images,is_train:False}) 
                #dev_disc_costs.append(_dev_disc_cost)
                wdtest=session.run(wd,feed_dict={is_train:False,real_data_int:images})
                dev_wdtest.append(wdtest)
            lib.plot.plot('mean mmd opt', np.mean(dev_mmd_costs))
            lib.plot.plot('mean wd test', np.mean(dev_wdtest))
            #lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % geniter == geniter-1):
            lib.plot.flush()

        lib.plot.tick()
