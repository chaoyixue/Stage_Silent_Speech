from utils.lsf_utils import *
import tensorflow as tf
#import soundfile as sf
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import argparse
from scipy import interpolate
class egg_Model():
    def __init__(self,cnn=True,dropout=0.1,shape=[200,100,50],output_size=266,input_size=266,activation=tf.nn.relu,
                 strides=1,kernel_size = [12,12,6],filters =[4,8,16],pool_size=[4,4,2],pooltype='max',fromhidden=False,optimize=True ):

        print('hello I can be 1d CNN / fully connected model from egg wave / sine wave / any wave to audio.')
        print("I can become better by parameters tuning, as Jianyu almost did not tune parameters.")
        self.input_size = input_size
        self.output_size = output_size
        self.input = tf.placeholder(shape=[None,self.input_size],dtype=tf.float32,name='input')
        self.output = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='output')
        if not cnn:
            hidden1 = tf.layers.dense(self.input,units=shape[0],activation=activation)
            hidden1 = tf.layers.dropout(hidden1,rate=dropout)
            hidden2 = tf.layers.dense(hidden1,units=shape[1],activation=activation)
            hidden2 = tf.layers.dropout(hidden2,rate=dropout)
            hidden3 = tf.layers.dense(hidden2,units=shape[2],activation=tf.nn.tanh)
            self.encoded = hidden3
            hidden3 = tf.layers.dropout(hidden3, rate=dropout)
            decode1 = tf.layers.dense(hidden3,units=shape[-2],activation=activation)
            decode1 = tf.layers.dropout(decode1,rate=dropout)
            decode2 = tf.layers.dense(decode1,units=shape[-3],activation=activation)
            decode2 = tf.layers.dropout(decode2,rate=dropout)
            decode3 = tf.layers.dense(decode2,units=self.input_size,activation=None)

            self.decode = decode3

        else:
            self.input = tf.placeholder(shape=[None, self.input_size,1], dtype=tf.float32, name='input')
            self.output = tf.placeholder(shape=[None, self.output_size,1], dtype=tf.float32, name='output')
            hidden1 = tf.layers.conv1d(self.input, filters=filters[0], kernel_size=kernel_size[0], strides=strides,
                             padding='same', activation=activation, use_bias=True)
            hidden1 = tf.layers.max_pooling1d(hidden1,pool_size=pool_size[0],strides=strides*2,padding='same')
            hidden1 = tf.layers.dropout(hidden1, rate=dropout)

            hidden2 = tf.layers.conv1d(hidden1, filters=filters[1], kernel_size=kernel_size[1], strides=strides,
                                       padding='same', activation=activation, use_bias=True)
            hidden2 = tf.layers.max_pooling1d(hidden2, pool_size=pool_size[1], strides=strides*2, padding='same')
            hidden2 = tf.layers.dropout(hidden2, rate=dropout)

            hidden3 = tf.layers.conv1d(hidden2, filters=filters[2], kernel_size=kernel_size[2], strides=strides,
                                       padding='same', activation=activation, use_bias=True)
            hidden3 = tf.layers.max_pooling1d(hidden3, pool_size=pool_size[2], strides=strides*2, padding='same')
            hidden3 = tf.layers.dropout(hidden3, rate=dropout)

            decode1 = tf.layers.conv1d(hidden3, filters=filters[2], kernel_size=kernel_size[2], strides=strides,
                                       padding='same', activation=activation, use_bias=True)
            shape = decode1.shape[1]

            decode1 = tf.reshape(decode1,[-1,1,shape,decode1.shape[-1]])
            decode1 = tf.image.resize_bilinear(decode1, (1,60))
            decode1 = tf.reshape(decode1, [-1, 60, decode1.shape[-1]])
            decode1 = tf.layers.dropout(decode1, rate=dropout)

            decode2 = tf.layers.conv1d(decode1, filters=filters[1], kernel_size=kernel_size[1], strides=strides,
                                       padding='same', activation=activation, use_bias=True)

            decode2 = tf.reshape(decode2, [-1, 1, decode2.shape[1], decode2.shape[-1]])
            decode2 = tf.image.resize_bilinear(decode2, (1,120))
            decode2 = tf.reshape(decode2,[-1,120,decode2.shape[-1]])
            decode2 = tf.layers.dropout(decode2, rate=dropout)

            decode3 = tf.layers.conv1d(decode2, filters=filters[0], kernel_size=kernel_size[0], strides=strides,
                                       padding='same', activation=activation, use_bias=True)

            decode3 = tf.reshape(decode3,[-1,1,decode3.shape[1],decode3.shape[-1]])
            decode3 = tf.image.resize_bilinear(decode3, (1,266))
            decode3 = tf.reshape(decode3,[-1,266,decode3.shape[-1]])
            decode3 = tf.layers.dropout(decode3, rate=dropout)

            self.decode = tf.layers.conv1d(decode3,filters=1,kernel_size=3, strides=1,padding='same',activation=None)

            #print(self.decode.shape)

        self.loss = tf.losses.mean_squared_error(labels=self.output, predictions=self.decode)
        if optimize:
            self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

def main(inputtype,outdir = '../out/cnn_fzero2audio/',f0only=False,usefilter=False,
    noise=False,lsf=False,outputid=0):

    size = int(16000 * 1.0/60)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    data_dir = '../out/'

    # trainegg, testegg, trainy, testy = loadeggsong('../out/',usefilter=usefilter)

    # if inputtype == 'egg':
    #     train,test = trainegg,testegg
    # else:
    #     if inputtype == 'f0song':
    #         f = open(data_dir+'trainsongfeat_fit.pkl','rb')
    #         trainegg = pickle.load(f)
    #         f.close()
    #         f = open(data_dir+'testsongfeat_fit.pkl','rb')
    #         testegg = pickle.load(f)
    #         f.close()
    #     elif inputtype == 'f0egg':
    #         f = open(data_dir + 'traineggfeat_fit.pkl', 'rb')
    #         trainegg = pickle.load(f)
    #         f.close()
    #         f = open(data_dir + 'testeggfeat_fit.pkl', 'rb')
    #         testegg = pickle.load(f)
    #         f.close()
        # train = []
        # for val in trainegg:
        #     if f0only:
        #         train.append(test_func(range(size),1,val[1],val[2],0))
        #     else:
        #         train.append(test_func(range(size),val[0],val[1],val[2],val[3]))

    f = open('../out/lipstongue2Fzero_traineach2/predict_index.pkl','rb')
    audioindex = pickle.load(f)
    f.close()  

    f = open('../out/lipstongue2Fzero_traineach2/lsf0.pkl','rb')
    T = pickle.load(f).flatten()
    
    uv_f = open("../out/lipstongue2uv/predict.pkl",'rb')
    testuvid = pickle.load(uv_f).argmax(axis=1).flatten()
    testuvid = -testuvid + 1
    def fixphase(A,T,D):

        phase=0
        phases=[0]
        size = len(A)
        for i in range(1,size):
            func = test_func2(A[i],T[i],D[i])
            pre_wave = test_func(np.array(range(256,266)),A[i-1],T[i-1],phase,D[i-1])
            losses =[]
            for j in range(200):
                now_wave = func(np.array(range(-11,-1)),j*1.0/100)
                loss = ((pre_wave - now_wave)**2).sum()
                losses.append(loss)
            phase = (np.array(losses).argmin()) * 1.0 / 100
            phases.append(phase)
        return np.array(phases)


    def feat2wave(A,T,P,D):
        waves = []
        for i in range(len(A)):
            wave = test_func(np.array(range(266)),A[i],T[i],P[i],D[i])
            waves.append(wave)
        return np.array(waves)

    allT = np.array([99999999999]*5000)
    allT[audioindex] = T
    allT[testuvid==0] = 99999999999
    print('#zero',(allT==0).sum())
    phase  = fixphase([1]*5000,allT,[0]*5000)

    test = feat2wave([1]*5000,allT,phase,[0]*5000)
    test = np.array(test)
    

    if noise:
    
        print('loading pred lsf 16khz')
        # print("loading pred lsf 16khz")
        lsf_f = open('../out/48746/predict_is16sd3.796.pkl','rb')

        testlsf = pickle.load(lsf_f)
        test_afterlsf_nonoise = vocoder(testlsf,test.flatten(),frame_pixel=266)
        testcopy = np.copy(test)
        for i in range(len(testuvid)):
            f=interpolate.interp1d(range(0,268,2),0.1*(np.random.randn(len(range(0,268,2)))-0.5),kind='slinear')
            if testuvid[i] == 0:
                test[i] = f(range(266))
            else:
                #f = interpolate.interp1d(range(0,300,6),0.1*(np.random.randn(len(range(0,300,6)))-0.5),kind='slinear')
                test[i] += f(range(266))
    
    # plt.plot(range(len(test.flatten())),test.flatten())
    # plt.show()   
    if lsf:
        print('loading pred lsf 16khz')
        # print("loading pred lsf 16khz")
        lsf_f = open('../out/48746/predict_is16sd3.796.pkl','rb')
        lsf = pickle.load(lsf_f)[0]
        trainlsf = np.concatenate([lsf[:25000],lsf[30000:]])
        testlsf = np.array(lsf[25000:30000])

        train_afterlsf  = vocoder(trainlsf,train.flatten(),frame_pixel=266)
        train = train_afterlsf.reshape((len(train),266))

        test_afterlsf  = vocoder(testlsf,test.flatten(),frame_pixel=266)
        test = test_afterlsf.reshape((len(test),266))


    test = test[:, :, np.newaxis]
    # cnn =True
    # if cnn:
    #     train = train[:,:,np.newaxis]
    #     test = test[:, :, np.newaxis]
    #     trainy = trainy[:, :, np.newaxis]
    #     testy = testy[:, :, np.newaxis]


    model = egg_Model(cnn=True,dropout=0)

    f = open(outdir + 'artificial.pkl','wb')
    pickle.dump(test.flatten(),f)
    f.close()

    # sf.write(outdir + 'artificial.wav',test.flatten(),16000,subtype='FLOAT')
    
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size=512

        # for i in range(1000000):

        #     batchid = np.random.randint(0, len(train), batch_size)
        #     batch_x = train[batchid]
        #     batch_y = trainy[batchid]
        #     sess.run(model.optimizer,feed_dict={model.input:batch_x,model.output:batch_y})
        #     if (i+1) % 1000 == 0:
        #         trainloss = sess.run(model.loss,feed_dict={model.input:batch_x,model.output:batch_y})
        #         testloss = sess.run(model.loss,feed_dict={model.input:test,model.output:testy})
        #         print("%d,trainloss %.6f, testloss %.6f" % (i+1,trainloss,testloss))
        #         pred = sess.run(model.decode,feed_dict={model.input:test})
        #         print(pred.max(),pred.min())
        #         f = open(outdir + 'ae_test+%d.pkl' %(i+1),'wb')
        #         pickle.dump(pred.flatten(),f)
        #         f.close()

        #         #sf.write(outdir+'ae_test_%d.wav' % (i+1),pred.flatten(),16000,subtype='FLOAT')

        #         plt.plot(range(len(test.flatten())),test.flatten(),label='origin',linewidth=0.01)
        #         plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=0.01)
        #         saver = tf.train.Saver()
        #         saver.save(sess,outdir+'model_%d.cpkt' % (i+1))
        #         plt.legend()
        #         plt.savefig(outdir+'ae_test_%d.png' % (i+1))
        #         plt.close()


        saver = tf.train.Saver()
        saver.restore(sess,outdir+'model_480000.cpkt')

        #testloss = sess.run(model.loss,feed_dict={model.input:test,model.output:testy})
        #print(testloss)
        pred = sess.run(model.decode,feed_dict={model.input:test})
        print(pred.max(),pred.min())


        wave = pred.flatten()

        # data *= (2**15 - 1)
        # data = data.astype(np.short)
        # f = wave.open(des, "wb")
        # f.setnchannels(1)
        # f.setsampwidth(2)
        # f.setframerate(default_fs)
        # f.writeframes(data.tostring())

        sf.write(outdir+'ae_test_480000_test_0.1noise.wav', wave / max(-wave.min(),wave.max()),16000,subtype='FLOAT')

        plt.plot(range(len(testy.flatten())),testy.flatten(),label='origin',linewidth=1)
        plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=1)
        plt.plot(range(len(test_afterlsf)), test_afterlsf, label='pred_lsf', linewidth=1)
        
        plt.xlabel('pixel')
        plt.legend()
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Run.')

    parser.add_argument('--inputtype', type=str,
                        default='f0song', help='from original egg or amplitude, cycle, phase, \
                        vertical shift from egg or amplitude, cycle, phase, \
                        vertical shift from audio to predict original audio. \
                        "egg","f0egg","f0song", by default we choose "f0song".')
    parser.add_argument('--f0only',type=int,default=1,help='use f0 only or not.')
    parser.add_argument('--usefilter',type=int,default=1,help='use filter or not.')

    return parser.parse_args()

if __name__ == "__main__":
    # args = parse_args()
    # usefilter = True if args.usefilter == 1 else False
    # f0only = True if args.f0only==1 else False
    # outdir = '../out/cnn_fzero2audio%s/' % ('f0only' if f0only else '')
    # main(args.inputtype,outdir=outdir, f0only=f0only,usefilter = usefilter)
    args = parse_args()
    usefilter = True if args.usefilter == 1 else False
    f0only = True if args.f0only==1 else False
    lsf = True
    noise = True 
    outdir = '../out/cnn_fzero2audio%s_noiselsf/' % ('f0only' if f0only else '')

    main(args.inputtype,outdir=outdir, f0only=f0only,usefilter = usefilter,noise=noise,lsf=lsf,outputid=5, )




