    
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from keras.layers import Activation
from keras.models import Model
import keras.layers as KL
import keras.backend as K
import numpy as np
from keras.utils.vis_utils import plot_model
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.losses import mean_squared_error,categorical_crossentropy
print(os.getcwd())
import tensorflow as tf
import random
os.environ ['PYTHONHASHSEED'] ='0'
np.random.seed(42)
random.seed(12345)
tf.set_random_seed(1234)
from keras.datasets import mnist 

#我开始以为是子集计算的溢出问题，但是我经过测试发现不是


def squaredDistance(X,Y):
    # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances
    # broadcasted subtraction, a square, and a sum.
    r = K.expand_dims(X, axis=1)
    return K.sum(K.square(r-Y), axis=-1)
        
    #calculate the raphy kernel applied to all entries in a pairwise distance matrix
def GaussianKernel(X,Y,scales):
    #res=np.zeros((X.get_shape().as_list()[0],Y.get_shape().as_list()[0]))
    sQdist = K.expand_dims(squaredDistance(X,Y),0) 
    #expand dist to a 1xnxm tensor where the 1 is broadcastable
    #expand sigma_list into a px1x1 tensor so we can do an element wise exponential
    scales = K.expand_dims(K.expand_dims(scales,-1),-1)
    #expand sigma_list into a px1x1 tensor so we can do an element wise exponential
    #calculated the kernal for each scale weight on the distance matrix and sum them up
    return K.sum(K.exp(-sQdist*(K.pow(scales,1))),0)##XJ:: 仔细查看Bioinformatic那篇文章的loss,你写的有问题

#Out[39]: <tf.Tensor 'Sum_4:0' shape=(3, 3) dtype=float32>,gaussian_kernel()返回的结果
    #calculated the kernal for each scale weight on the distance matrix and sum them up
    #return K.sum(self.weights*K.exp(-sQdist / (K.pow(self.scales,2))),0)

#Calculate the MMD cost
def cost(source, target,sigma_list):
    MMD=0;

    tempxy=K.zeros_like(GaussianKernel(source,target,2.0))
    tempxx=K.zeros_like(GaussianKernel(source,source,2.0))
    tempyy=K.zeros_like(GaussianKernel(target,target,2.0))
    #tempxx=tf.placeholder(tf.float32, [None, None])
    #tempyy=tf.placeholder(tf.float32, [None, None])
    #calculate the 3 MMD terms
    #return the square root of the MMD because it optimizes better
    #return K.sqrt(MMD);
    for scales in sigma_list: 
        scale = 1.0 / (2 * scales**2)
        xy =GaussianKernel(source, target,scale)
        tempxy=tf.add(tempxy,xy)
        xx =GaussianKernel(source, source,scale)#注意先传入的是X和X，与Y无关，那么SQdist，返回的是X之间的距离
        tempxx=tf.add(tempxx,xx)
        yy =GaussianKernel(target, target,scale)
        tempyy=tf.add(tempyy,yy)
        #tempyy=tf.add(tempyy,yy)
    #calculate the bias MMD estimater (cannot be less than 0)
    
    MMD = K.mean(tempxx) - 2 * K.mean(tempxy) + K.mean(tempyy)
    #return the square root of the MMD because it optimizes better
    return (MMD);
#上面写的代码都是为了计算MMD_loss


sess=tf.InteractiveSession()
n_batch=2
labda=9
base = 1.0
sigma_list = [1,2,4]#
sigma_list = [sigma / base for sigma in sigma_list]

#==================问题所在==============================#
#修改下面两个值，看训练结果
#这个地方确实有问题
sample_size=190
batch_size=200;
#如果sample_size>batch_size,则val_loss 出现nan
#如果sample_size<=batch_size,则val_loss 会显示具体的数值
#================================================#


#产生测试数据
x_train=np.array([]).reshape((0,1000))
y_train=np.array([],dtype="int32").reshape((0,))
x_test=np.array([]).reshape((0,1000))
y_test=np.array([],dtype="int32").reshape((0,))

for i in range(0,2):
    temp_x=np.random.normal(loc=i,scale=1,size=(6000,1000))
    test_x=np.random.normal(loc=i,scale=1,size=(sample_size,1000))
    x_train=np.concatenate((x_train,temp_x))
    x_test=np.concatenate((x_test,test_x))
    y_train=np.concatenate((y_train,np.full((600,),i,dtype="int32")))
    y_test=np.concatenate((y_test,np.full((sample_size,),i,dtype="int32")))

#将数据按类排成字典的形式
trainDataLoader={}
trainData=np.array([]).reshape((0,1000))
train_label=np.array([],dtype="int32").reshape((0,))
testDataLoader={}
testData=np.array([]).reshape((0,1000))
test_label=np.array([],dtype="int32").reshape((0,))

for i in range(0,2):
    trainDataLoader[i]=x_train[y_train==i]
    trainData=np.concatenate((trainData,trainDataLoader[i]))
    train_label=np.concatenate((train_label,np.full((trainDataLoader[i].shape[0],),i,dtype="int32")))
    testDataLoader[i]=x_test[y_test==i]
    testData=np.concatenate((testData,testDataLoader[i]))
    test_label=np.concatenate((test_label,np.full((testDataLoader[i].shape[0],),i,dtype="int32")))

y_train=np.reshape(y_train,(-1,1))
y_test=np.reshape(y_test,(-1,1))
test_label=np.reshape(test_label,(-1,1))

print("testData.shape=",testData.shape)
print("test_label.shape",test_label.shape)

input_x = KL.Input(shape=(1000,),name="input_x")
input_y_class = KL.Input(shape=(1,),name="input_y_class")
x1 = KL.Dense(256, activation='relu')(input_x)
encode=KL.Dense(32, activation='linear')(x1)
decode=KL.Dense(256,activation='relu')(encode)
output=KL.Dense(1000,activation="linear",name="output")(decode)


def generator():
    while 1:
        data_x=np.array([]).reshape((0,1000));
        data_y=np.array([],dtype="int32").reshape((0,1));
        for i in range(0,n_batch):
            row = np.random.randint(0,trainDataLoader[i].shape[0],size=batch_size)
            temp_x=trainDataLoader[i][row]#直接从某个类中选取样本
            temp_y=np.full((temp_x.shape[0],1),i,dtype="int32")            
            data_x=np.concatenate((data_x,temp_x))
            data_y=np.concatenate((data_y,temp_y))
        yield [data_x,data_y],None#
        
def mmd_loss(y_label,encode):
    num=[]
    for i in range(0,n_batch):
        num.append(K.sum(K.cast(tf.equal(y_label,i),dtype="int32")))
    z=tf.split(encode,num,axis=0)

    mmd=0.0;
    for z1 in z:
        for z2 in z:
            mmd+=cost(z1,z2,sigma_list) 
    mmd=mmd*2/(n_batch*(n_batch-1))
    #return K.mean(mmd)# 
    return mmd

def cons_loss(input_layer,output):
    return K.mean(mean_squared_error(input_layer,output))
def total_loss(mmd_loss,cons_loss):
    return mmd_loss+labda*cons_loss
mmd_loss=KL.Lambda(lambda x:mmd_loss(*x),name='mmd_loss')([input_y_class,encode])
cons_loss=KL.Lambda(lambda x:cons_loss(*x),name='cons_loss')([input_x,output])
total_loss=KL.Lambda(lambda x:total_loss(*x),name='total_loss')([mmd_loss,cons_loss])


model = Model([input_x,input_y_class], [total_loss])
model.get_layer('mmd_loss').output
model.add_loss(total_loss)

# encode_model=Model(input_x,encode)
# plot_model(model,"two_batch_total_model_Retina.png",show_shapes=True)

# #mmd_model
# MMD_model=K.function([model.input],[model.get_layer('mmd_loss').output])
# model_mmd=Model(inputs=model.input,outputs=model.get_layer('mmd_loss').output)
# plot_model(model_mmd,"two_batch_mmd_model_Retina.png",show_shapes=True)#使用K.function是不能画图的，如果想画子模型的图，还是得要

# #reconstruction_model
# reconstruction_model=K.function([model.get_layer("input_x").input],[model.get_layer("cons_loss").output])
# model_reconstruction=Model(inputs=model.get_layer("input_x").input,outputs=model.get_layer("cons_loss").output)
# plot_model(model_reconstruction,"two_batch_reconstruction_model_Retina.png",show_shapes=True)

pretrain_optimizer=keras.optimizers.Adam(lr =0.001 , beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer=pretrain_optimizer)

history = model.fit_generator(generator(),epochs=2,steps_per_epoch=3,validation_data=([testData,test_label],None))
sess.close()