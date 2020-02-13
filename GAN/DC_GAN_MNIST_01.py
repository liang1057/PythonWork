'''
生成对抗网络（Generative Adversarial Networks，GAN）
最早由 Ian Goodfellow 在 2014 年提出，是目前深度学习领域最具潜力的研究成果之一
它的核心思想是：同时训练两个相互协作、同时又相互竞争的深度神经网络（一个称为生成器 Generator，另一个称为判别器 Discriminator）
来处理无监督学习的相关问题。在训练过程中，两个网络最终都要学习如何处理任务。

本文将以深度卷积生成对抗网络（Deep Convolutional GAN，DCGAN）为例，介绍如何基于 Keras 2.0 框架，以 Tensorflow 为后端，
搭建一个真实可用的 GAN 模型，并以该模型为基础自动生成 MNIST 手写体数字
'''


'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
#from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils  # to_categorical OneHot

import matplotlib.pyplot as plt

# 划分训练集和测试集的样本比例
def divSamples(x_all, y_all, per=0.8):
    index = [i for i in range(len(x_all))]
    np.random.shuffle(index)
    trainIndex = index[0: (int)(len(x_all)*per)]
    trainIndex = np.array(trainIndex)
    x_train = x_all[trainIndex]
    y_train = y_all[trainIndex]

    testIndex = index[(int)(len(x_all)*per) : len(x_all)]
    testIndex = np.array(testIndex)
    x_test = x_all[testIndex]
    y_test = y_all[testIndex]
    return (x_train, y_train),(x_test, y_test)

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.inputsize = 10
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def lockD(self, lock=True):
        if lock == True:
            for layer in self.D.layers:
                layer.trainable = False
        else:
            for layer in self.D.layers:
                layer.trainable = True

    # (W−F+2P)/S+1
    '''判别器：构造一个CNN，用来识别一个图片.'''
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        '''
        depth = 16 #64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, \
                          padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Dropout(dropout))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        '''
        self.D = Sequential()
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Convolution2D(16, kernel_size=(3,3), input_shape=input_shape,
                                 padding='same', activation='relu',
                                 name='conv2D_1')) # 层的名称，可以用来获取该层的执行情况
        self.D.add(Dropout(0.25)) #随机失活

        # activation 也可以单独写出来作为一层
        self.D.add(Convolution2D(16, kernel_size=(3,3))) #卷积层
        self.D.add(Activation("relu"))
        self.D.add(Dropout(0.25)) #随机失活

        self.D.add(MaxPooling2D(pool_size=(2,2))) #池化层

        self.D.add(Convolution2D(16, kernel_size=(3,3))) #卷积层
        self.D.add(Activation("relu"))
        self.D.add(Dropout(0.25)) #随机失活

        self.D.add(Flatten()) #拉成一维数据

        self.D.add(Dense(128)) #全连接层1
        self.D.add(Activation('relu')) #激活层
        self.D.add(Dropout(0.25)) #随机失活

        self.D.add(Dense(11))
        self.D.add(Activation('sigmoid'))

        print("D is OK")
        return self.D

    '''生成器，这个是用来生成对抗的, 将一组随机数生成一个新图片'''
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()

        depth = 128
        n_dropout = 0.25
        self.G.add(Dense(7*7*depth, input_dim=self.inputsize))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((7, 7, depth)))
        self.G.add(Dropout(n_dropout))

        # 一次上采样，变成 2dim * 2dim * depth ：14*14
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 3, padding='same')) # 逆卷积
        #self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(n_dropout))

        self.G.add(UpSampling2D())  # 二次上采样，变成 28*28
        self.G.add(Conv2DTranspose(int(depth/4), 3, padding='same')) # 逆卷积
        #self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(n_dropout))

        self.G.add(Conv2DTranspose(int(depth/8), 3, padding='same')) # 逆卷积
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(n_dropout))

        #self.G.add(Reshape((28, 28, 4)))
        self.G.add(Conv2DTranspose(1, 3, padding='same'))
        #self.G.add(BatchNormalization(momentum=0.9))
        #self.G.add(Dropout(0.5))
        self.G.add(Activation('sigmoid'))
        return self.G


    def discriminator_model(self):
        '''
        分辨器模型，用来分辨给定的样本
        :return:
        '''
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0001, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # LR = 0.0001  #学习率
        # adam = Adam(LR)
        # self.DM.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        '''
        对抗模型，用来生成一个新的样本，该样本使得分辨器无法做出判断
        :return:
        '''
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,  metrics=['accuracy'])
        # adam = Adam(0.0001)
        # self.AM.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return self.AM



class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        (X_train, y_train), (X_test, y_test) = mnist.load_data() # 'X_train.shape = ', (60000, 28, 28)

        self.x_train = X_train.astype('float32')/ 255.
        self.y_train = y_train
        self.x_test = X_test.astype('float32')/ 255.
        self.y_test = y_test

        # self.x_train = input_data.read_data_sets("mnist", \
        #                                          one_hot=True).train.images
        # self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
        self.x_train = self.x_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        self.x_test = self.x_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

        self.DCGAN = DCGAN()
        self.DCGAN.inputsize = 110
        self.discriminator =  self.DCGAN.discriminator_model()  # 没有的时候要现训练。这里用已有的网络
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    '''
    根据给定的 tNum 数字，生成一个随机向量。
    该随机向量用于给生成器当作输入（输入一个近似于one-hot的随机向量）
    '''
    def makeRandomVec(self, tNum):
        temp = np_utils.to_categorical(tNum, num_classes = 10).reshape(10,)
        tempRand = np.random.rand(100).reshape(100,)
        tSample = np.concatenate((tempRand, temp))
        return tSample
        # temp = np_utils.to_categorical(tNum, num_classes = 10)
        # tempRand = np.random.rand(10)
        # tSample = temp + 0.1 * tempRand
        # return tSample

    def train(self, train_steps=2000, batch_size=512, save_interval=0):
        tIndex = np.random.randint(0, self.x_train.shape[0], size=batch_size)
        print('tIndex.shape = ', tIndex.shape)
        inputsize = 110
        # tIndex16 = np.random.uniform(0, 9, size=[16, 1]).astype('int32')
        # self.plot_images2(tIndex16)

        fake_number = 16 # 随机模拟的数字的个数
        noise_input = np.random.uniform(0, 1.0, size=[fake_number, inputsize]) #      显示的16个数字的输入
        noise_Num = np.random.uniform(0, 9, size=[fake_number, 1]).astype('int32') # 显示的16个数字的值
        for i in range(0,10):
            noise_Num[i] = i
        for i in range(0, fake_number):
            noise_input[i,:] = self.makeRandomVec(noise_Num[i])  # 重新设置随机数值

        print('Numbers: ', noise_Num)  # 要打印出的 用于对抗生成的数字

        ret_y_train = np_utils.to_categorical(self.y_train, num_classes = 11) # to OneHot
        ret_y_test = np_utils.to_categorical(self.y_test, num_classes = 11)
        for i in range(train_steps):  # 每次，相当于echop
            print('\n训练EPOCH = ', i)

            tSize_real = self.y_train.shape[0]
            tSize_fake = (int)(tSize_real * 0.25) # 伪造的数量为真样本的25%
            
            #为了好比对，随机抽取真实的Y集，将类别标签给伪造序列，伪造的数据在分类器中进行识别的时候就按这些标签，让伪造数据试图骗过分类器
            tIndex = np.random.randint(0, tSize_real, size=tSize_fake)
            randY = self.y_train[tIndex]
            
            discY = 10 * np.ones(tSize_fake, dtype='int32') # 分辨器认为这些伪造的类型是10

            randX = np.zeros((tSize_fake, inputsize), dtype="float32")
            for j in range(0, tSize_fake):
                randX[j,:] = self.makeRandomVec(randY[j])

            images_fake = self.generator.predict(randX)  # 生成器生成的图片
            images_fake.reshape(images_fake.shape[0], images_fake.shape[1], images_fake.shape[2], 1)
            x = np.concatenate((self.x_train, images_fake))    #将给定的图片与伪造的图片合并成一个数据集，作为训练分类器的输入
            disc_y = np.concatenate((ret_y_train, np_utils.to_categorical(discY, num_classes = 11))) #其中伪造的标签为10

            ret_randY = np_utils.to_categorical(randY, num_classes = 11) #
            ret_fakeY = np_utils.to_categorical(discY, num_classes = 11) # 分类器的标准输出（训练D）
            (x_train, y_train),(x_test, y_test) = divSamples(x, disc_y, 0.5)  # 训练数据用于训练D

            # print('训练D, 让分类器能够将伪造的图片识别出来')
            self.discriminator.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(images_fake, ret_fakeY))  #伪造数据的标签为10
            ret = self.adversarial.predict(noise_input)
            ret_max = np.argmax(ret, axis = 1)
            # print('orignal: ',noise_Num)
            print('训练D, 让分类器能够将伪造的图片识别出来, result:  ',ret_max)

            self.DCGAN.lockD(True) # 锁定判别器，只训练生成器
            # print('训练G+D，让生成器伪造的图片能够正确分类（骗过分类器）')
            self.adversarial.fit(randX, ret_randY, batch_size=64, epochs=1, verbose=1, validation_data=(randX, ret_randY))
            self.DCGAN.lockD(False)
            ret = self.adversarial.predict(noise_input)
            ret_max = np.argmax(ret, axis = 1)
            print('训练G+D，让生成器伪造的图片能够正确分类（骗过分类器）,result:  ',ret_max)

            if save_interval>0:
                self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))
                self.saveModel(tPath+'/model/gan_G_%d.h5'%(i+1),  tPath + '/model/gan_D_%d.h5'%(i+1))


    def saveModel(self, fileG, fileD):
        print('save G: ', fileG)
        self.generator.save(fileG)
        print('save D: ', fileD)
        self.discriminator.save(fileD)

    def loadModel(self, fileG, fileD):
        self.generator = load_model(fileG)
        # self.generator.compile()
        optimizer = RMSprop(lr=0.0001, decay=6e-8)
        LR = 0.0001  #学习率
        adam = Adam(LR)
        self.DCGAN.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.discriminator = load_model(fileD)
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        #self.discriminator.compile()
        self.DCGAN.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = ''
        if fake:
            g_input = noise
            if noise is None:
                #noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
                noise = np.random.uniform(0, 1.0, size=[samples, 128])
                g_input = g_input #np.zeros((16, 128), dtype="float32")
                # for tNum in range(0, noise.shape[0]):
                #     g_input[tNum,:] = np_utils.to_categorical(noise[tNum], num_classes = 11)
            else:
                filename = tPath + "/image/mnist_%d.png" % step

            images = self.generator.predict(g_input)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    # def plot_images2(self, tIndex16):
    #     filename = './notes/Gan-mnist-16.png'
    #     images = np.array((16, self.img_rows, self.img_cols, 1), dtype='float32')
    #     for i in range(16):
    #         images(self.x_train[tIndex16[i], :, :, :])
    #
    #     plt.figure(figsize=(10,10))
    #     for i in range(images.shape[0]):
    #         plt.subplot(4, 4, i+1)
    #         image = images[i, :, :, :]
    #         image = np.reshape(image, [self.img_rows, self.img_cols])
    #         plt.imshow(image, cmap='gray')
    #         plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(filename)
    #     plt.close('all')


if __name__ == '__main__':
    tPath = './notes/note2'
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=200, batch_size=128, save_interval=5)
    mnist_dcgan.saveModel(tPath+'/model/final_G.h5', tPath+'/model/final_G.h5')
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)

