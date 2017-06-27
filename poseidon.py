import os
import subprocess
import sys
import numpy as np
import random
from copy import copy
import theano
from theano import tensor as T

import blocks
from blocks.bricks import Linear, Softmax, Softplus, NDimensionalSoftmax, BatchNormalizedMLP,\
                          Rectifier, Logistic, Tanh, MLP
from blocks.bricks.recurrent import GatedRecurrent, Fork, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Identity, Uniform
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER
from blocks.graph import ComputationGraph

import logging
import cPickle
import sessionizer

import learningfunctions
import adversarialfunctions

import theano.d3viz as d3v


#load and save files and models
def pickleFile(thing2save, file2save2 = None, filePath='/work/notebooks/drawModels/', fileName = 'myModels'):

    if file2save2 == None:
        f=file(filePath+fileName+'.pickle', 'wb')
    else:
        f=file(filePath+file2save2, 'wb')

    cPickle.dump(thing2save, f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()

def loadFile(filePath):
    file2open = file(filePath, 'rb')
    loadedFile = cPickle.load(file2open)
    file2open.close()

    return loadedFile

#Making the hex dictionary
def hexTokenizer():
    hexstring = '0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	A,	B,	C,	D,	E,	F,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19\
    ,	1A,	1B,	1C,	1D,	1E,	1F,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	2A,	2B,	2C,	2D,	2E,	2F,	30,	31,	32,	33,	34,	35\
    ,	36,	37,	38,	39,	3A,	3B,	3C,	3D,	3E,	3F,	40,	41,	42,	43,	44,	45,	46,	47,	48,	49,	4A,	4B,	4C,	4D,	4E,	4F,	50,	51\
    ,	52,	53,	54,	55,	56,	57,	58,	59,	5A,	5B,	5C,	5D,	5E,	5F,	60,	61,	62,	63,	64,	65,	66,	67,	68,	69,	6A,	6B,	6C,	6D\
    ,	6E,	6F,	70,	71,	72,	73,	74,	75,	76,	77,	78,	79,	7A,	7B,	7C,	7D,	7E,	7F,	80,	81,	82,	83,	84,	85,	86,	87,	88,	89\
    ,	8A,	8B,	8C,	8D,	8E,	8F,	90,	91,	92,	93,	94,	95,	96,	97,	98,	99,	9A,	9B,	9C,	9D,	9E,	9F,	A0,	A1,	A2,	A3,	A4,	A5\
    ,	A6,	A7,	A8,	A9,	AA,	AB,	AC,	AD,	AE,	AF,	B0,	B1,	B2,	B3,	B4,	B5,	B6,	B7,	B8,	B9,	BA,	BB,	BC,	BD,	BE,	BF,	C0,	C1\
    ,	C2,	C3,	C4,	C5,	C6,	C7,	C8,	C9,	CA,	CB,	CC,	CD,	CE,	CF,	D0,	D1,	D2,	D3,	D4,	D5,	D6,	D7,	D8,	D9,	DA,	DB,	DC,	DD\
    ,	DE,	DF,	E0,	E1,	E2,	E3,	E4,	E5,	E6,	E7,	E8,	E9,	EA,	EB,	EC,	ED,	EE,	EF,	F0,	F1,	F2,	F3,	F4,	F5,	F6,	F7,	F8,	F9\
    ,	FA,	FB,	FC,	FD,	FE,	FF'.replace('\t', '')

    hexList = [x.strip() for x in hexstring.lower().split(',')]
    hexList.append('<EOP>') #End Of Packet token
    #EOS token??????
    hexDict = {}

    for key, val in enumerate(hexList):
        if len(val) == 1:
            val = '0'+val
        hexDict[val] = key  #dictionary: k=hex, v=int

    return hexDict


def srcIpDict(hexSessionDict):
    '''
    input: dictionary of key = sessions, value = list of HEX HEADERS of packets in session
    output: dictionary of key = source IP, value/subkey = dictionary of destination IPs,
                                           subvalue = [[sport], [dport], [plen], [protocol]]

    '''

    srcIpDict = {}
    uniqIPs = [] #some ips are dest only. this will collect all ips, not just srcIpDict.keys()

    for session in hexSessionDict.keys():

        for rawpacket in hexSessionDict[session][0]:
            packet = copy(rawpacket)

            dstIpSubDict = {}

            sourceMAC = packet[:12]
            destMAC = packet[12:24]
            srcip = packet[52:60]
            dstip = packet[60:68]
            sport = packet[68:72]
            dport = packet[72:76]
            plen = packet[32:36]
            protocol = packet[46:48]

            uniqIPs = list(set(uniqIPs) | set([dstip, srcip]))

            if srcip not in srcIpDict:
                dstIpSubDict[dstip] = [[sport], [dport], [plen], [protocol], [sourceMAC], [destMAC]]
                srcIpDict[srcip] = dstIpSubDict

            if dstip not in srcIpDict[srcip]:
                srcIpDict[srcip][dstip] = [[sport], [dport], [plen], [protocol], [sourceMAC], [destMAC]]
            else:
                srcIpDict[srcip][dstip][0].append(sport)
                srcIpDict[srcip][dstip][1].append(dport)
                srcIpDict[srcip][dstip][2].append(plen)
                srcIpDict[srcip][dstip][3].append(protocol)
                srcIpDict[srcip][dstip][4].append(sourceMAC)
                srcIpDict[srcip][dstip][5].append(destMAC)

    return srcIpDict, uniqIPs

def oneHot(index, granular = 'hex'):
    if granular == 'hex':
        vecLen = 257
    else:
        vecLen = 17

    zeroVec = np.zeros(vecLen)
    zeroVec[index] = 1.0

    return zeroVec


def dictUniquerizer(dictOdictsOlistOlists):
    '''
    input is the output of srcIpDict
    input: dictionary of dictionaries that have a list of lists
           ex. srcIpDict[srcip][dstip] = [[sport], [dport], [plen], [protocol]]
    output: dictionary of dictionaries with list of lists with unique items in the final sublist

    WARNING: will overwrite your input dictionary. Make a copy if you want to preserve dictOdictsOlistOlists.
    '''
    #dictCopy
    for key in dictOdictsOlistOlists.keys():
        for subkey in dictOdictsOlistOlists[key].keys():
            for sublist in xrange(len(dictOdictsOlistOlists[key][subkey])):
                dictOdictsOlistOlists[key][subkey][sublist] = list(set(dictOdictsOlistOlists[key][subkey][sublist]))

    return dictOdictsOlistOlists

#TODO: add character level encoding
def oneSessionEncoder(sessionPackets, hexDict, maxPackets = 2, packetTimeSteps = 100,
                      packetReverse = False, charLevel = False, padOldTimeSteps = True,
                      onlyEssentials = False):

    sessionCollect = []
    packetCollect = []

    if charLevel:
        vecLen = 17
    else:
        vecLen = 257

    if len(sessionPackets) > maxPackets: #crop the number of sessions to maxPackets
        sessionList = copy(sessionPackets[:maxPackets])
    else:
        sessionList = copy(sessionPackets)

    for rawpacket in sessionList:
        packet = copy(rawpacket)

        if onlyEssentials: #cat of length,protocol,frag,srcIP,dstIP,srcport,dstport
            packet = packet[32:36]+packet[44:46]+packet[46:48]+packet[52:60]+packet[60:68]+\
                     packet[68:72]+packet[72:76]

        packet = [hexDict[packet[i:i+2]] for i in xrange(0,len(packet)-2+1,2)] #get hex pairs

        if len(packet) >= packetTimeSteps - 1: #crop packet to length packetTimeSteps rel to hex pairs
            packet = packet[:packetTimeSteps - 1]

        packet = packet+[256] #add <EOP> end of packet token. now have packetTimeSteps

        packetCollect.append(packet)

        pacMat = np.array([oneHot(x) for x in packet]) #one hot encoding of packet into a matrix
        pacMatLen = len(pacMat)

        #padding packet
        if packetReverse:
            pacMat = pacMat[::-1]

        if pacMatLen < packetTimeSteps:
            #pad by stacking zeros on top of data so that earlier timesteps do not have information
            #padding the packet such that zeros are after the actual info for better translation
            if padOldTimeSteps:
                pacMat = np.vstack( ( np.zeros((packetTimeSteps-pacMatLen,vecLen)), pacMat) )
            else:
                pacMat = np.vstack( (pacMat, np.zeros((packetTimeSteps-pacMatLen,vecLen))) )

        #if pacMatLen >= packetTimeSteps + 1:
        #    pacMat = pacMat[:packetTimeSteps, :]

        sessionCollect.append(pacMat)

    #padding session
    sessionCollect = np.asarray(sessionCollect, dtype=theano.config.floatX)
    numPacketsInSession = sessionCollect.shape[0]
    if numPacketsInSession < maxPackets:
        #pad sessions to fit the
        sessionCollect = np.vstack( (sessionCollect,np.zeros((maxPackets-numPacketsInSession,
                                                             packetTimeSteps, vecLen))) )

    return sessionCollect, packetCollect


#EVAL FUNCTION

def predictClass(predictFun, hexSessionsDict, comsDict, uniqIPs, hexDict, hexSessionsKeys,
                 binaryTarget, numClasses, onlyEssentials, trainPercent = 0.9, dimIn=257, maxPackets=2,
                 packetTimeSteps = 16, padOldTimeSteps=True):

    testCollect = []
    predtargets = []
    actualtargets = []
    trainIndex = int(len(hexSessionsKeys)*trainPercent)

    start = trainIndex
    end = len(hexSessionsKeys)

    trainingSessions = []
    trainingTargets = []

    for trainKey in range(start, end):
        sessionForEncoding = hexSessionsDict[hexSessionsKeys[trainKey]][0]

        adfun = adversarialfunctions.Adversary(sessionForEncoding)
        adversaryList = [sessionForEncoding,
                         adfun.dstIpSwapOut(comsDict, uniqIPs),
                         adfun.portDirSwitcher(),
                         adfun.ipDirSwitcher(),
                         adfun.noisyPacketMaker(maxPackets, packetTimeSteps, percentNoisy = 0.2)]
        if binaryTarget:
            # choose normal and one of the abnormal types
            abbyIndex = random.sample([0, random.sample(xrange(1,len(adversaryList)), 1)[0]], 1)[0]
            if abbyIndex == 0:
                targetClasses = [0,1]
            else:
                targetClasses = [1,0]
        else:
            assert len(adversaryList)==numClasses
            abbyIndex = random.sample(range(len(adversaryList)), 1)[0]
            targetClasses = [0]*numClasses
            targetClasses[abbyIndex] = 1

        abbyOneHotSes = oneSessionEncoder(adversaryList[abbyIndex],
                                          hexDict = hexDict,
                                          packetReverse=packetReverse,
                                          padOldTimeSteps = padOldTimeSteps,
                                          maxPackets = maxPackets,
                                          packetTimeSteps = packetTimeSteps,
                                          onlyEssentials = onlyEssentials)

        trainingSessions.append(abbyOneHotSes[0])
        trainingTargets.append(np.array(targetClasses, dtype=theano.config.floatX))

    sessionsMinibatch = np.asarray(trainingSessions, dtype=theano.config.floatX)\
                                   .reshape((-1, packetTimeSteps, 1, dimIn))
    targetsMinibatch = np.asarray(trainingTargets, dtype=theano.config.floatX)

    predcostfun = predictFun(sessionsMinibatch)
    testCollect.append(np.mean(np.argmax(predcostfun,axis=1) == np.argmax(targetsMinibatch, axis=1)))

    predtargets = np.argmax(predcostfun,axis=1)
    actualtargets = np.argmax(targetsMinibatch, axis=1)

    print "TEST accuracy:         ", np.mean(testCollect)
    print

    return predtargets, actualtargets, np.mean(testCollect)

def binaryPrecisionRecall(predictions, targets, numClasses = 4):
    for cla in range(numClasses):

        confustop = np.array([])
        confusbottom = np.array([])

        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()

        pred1 = np.where(predictions == cla)
        pred0 = np.where(predictions != cla)
        target1 = np.where(targets == cla)
        target0 = np.where(targets != cla)

        truePos = np.intersect1d(pred1[0],target1[0]).shape[0]
        trueNeg = np.intersect1d(pred0[0],target0[0]).shape[0]
        falsePos = np.intersect1d(pred1[0],target0[0]).shape[0]
        falseNeg = np.intersect1d(pred0[0],target1[0]).shape[0]

        top = np.append(confustop, (truePos, falsePos))
        bottom = np.append(confusbottom, (falseNeg, trueNeg))
        confusionMatrix = np.vstack((top, bottom))

        precision  = float(truePos)/(truePos + falsePos + 0.00001) #1 - (how much junk did we give user)
        recall = float(truePos)/(truePos + falseNeg + 0.00001) #1 - (how much good stuff did we miss)
        f1 = 2*((precision*recall)/(precision+recall+0.00001))

        print 'class '+str(cla)+' precision: ', precision
        print 'class '+str(cla)+' recall:    ', recall
        print 'class '+str(cla)+' f1:        ', f1



def onestepContext(hEncReshape):

    data3, data4 = forkContext.apply(hEncReshape)

    if rnnType == 'gru':
        hContext = rnnContext.apply(data3, data4)
    else:
        hContext, _ = rnnContext.apply(data4)

    return hContext

#UNSUPERVISED FEATURE EXTRACTOR
#INITIALIZATION UNSUPERVISED NET AND CLASSIFIER

def training(runname, rnnType, onlyEssentials, maxPackets, packetTimeSteps, packetReverse, padOldTimeSteps, wtstd,
             lr, decay, clippings, dimIn, dim, binaryTarget, numClasses, batch_size, epochs,
             trainPercent, dataPath, savePath, loadPrepedData = False):
    print locals()
    print

    X = T.tensor4('inputs')
    Y = T.matrix('targets')
    linewt_init = IsotropicGaussian(wtstd)
    line_bias = Constant(1.0)
    rnnwt_init = IsotropicGaussian(wtstd)
    rnnbias_init = Constant(0.0)
    classifierWts = IsotropicGaussian(wtstd)

    learning_rate = theano.shared(np.array(lr, dtype=theano.config.floatX))
    learning_decay = np.array(decay, dtype=theano.config.floatX)

    ###DATA PREP
    print 'loading data'
    if loadPrepedData:
        hexSessions = loadFile(dataPath)
    else:
        sessioner = sessionizer.Sessionizer(dataPath)
        #Dizionario con key = srcip:srcport,dstip:dstport
        hexSessions = sessioner.packetizer()

    hexSessions = sessionizer.removeBadSessionizer(hexSessions)
    hexSessionsKeys = sessioner.order_keys()
    hexDict = hexTokenizer()

    print 'creating dictionary of ip communications'
    comsDict, uniqIPs = srcIpDict(hexSessions)
    comsDict = dictUniquerizer(comsDict)

    print 'initializing network graph'
    ###ENCODER RNN
    if rnnType == 'gru':
        rnn = GatedRecurrent(dim=dim, weights_init = rnnwt_init, biases_init = rnnbias_init, name = 'gru')
        dimMultiplier = 2
    else:
        rnn = LSTM(dim=dim, weights_init = rnnwt_init, biases_init = rnnbias_init, name = 'lstm')
        dimMultiplier = 4

    fork = Fork(output_names=['linear', 'gates'],
                name='fork', input_dim=dimIn, output_dims=[dim, dim * dimMultiplier],
                weights_init = linewt_init, biases_init = line_bias)

    ###CONTEXT RNN
    if rnnType == 'gru':
        rnnContext = GatedRecurrent(dim=dim, weights_init = rnnwt_init,
                                    biases_init = rnnbias_init, name = 'gruContext')
    else:
        rnnContext = LSTM(dim=dim, weights_init = rnnwt_init, biases_init = rnnbias_init,
                          name = 'lstmContext')

    forkContext = Fork(output_names=['linearContext', 'gatesContext'],
                name='forkContext', input_dim=dim, output_dims=[dim, dim * dimMultiplier],
                weights_init = linewt_init, biases_init = line_bias)

    forkDec = Fork(output_names=['linear', 'gates'],
                name='forkDec', input_dim=dim, output_dims=[dim, dim*dimMultiplier],
                weights_init = linewt_init, biases_init = line_bias)

    #CLASSIFIER
    bmlp = BatchNormalizedMLP( activations=[Logistic(), Logistic()],
               dims=[dim, dim, numClasses],
               weights_init=classifierWts,
               biases_init=Constant(0.0001) )

    fork.initialize()
    rnn.initialize()
    forkContext.initialize()
    rnnContext.initialize()
    forkDec.initialize()
    bmlp.initialize()

    def onestepEnc(X):
        data1, data2 = fork.apply(X)
        #print data1.eval({X:[[1]]})
        #print data2.eval({X:[[1]]})

        if rnnType == 'gru':
            hEnc = rnn.apply(data1, data2)
        else:
            hEnc, _ = rnn.apply(data2)

        return hEnc

    hEnc, _ = theano.scan(onestepEnc, X)
    hEncReshape = T.reshape(hEnc[:,-1], (-1, maxPackets, 1, dim))

    def onestepContext(hEncReshape):

        data3, data4 = forkContext.apply(hEncReshape)

        if rnnType == 'gru':
            hContext = rnnContext.apply(data3, data4)
        else:
            hContext, _ = rnnContext.apply(data4)

        return hContext

    hContext, _ = theano.scan(onestepContext, hEncReshape)
    hContextReshape = T.reshape(hContext[:,-1], (-1,dim))

    data5, _ = forkDec.apply(hContextReshape)
    pyx = bmlp.apply(data5)
    softmax = Softmax()
    softoutClass = softmax.apply(pyx)
    costClass = T.mean(CategoricalCrossEntropy().apply(Y, softoutClass))

    #CREATE GRAPH
    cgClass = ComputationGraph([costClass])
    paramsClass = VariableFilter(roles = [PARAMETER])(cgClass.variables)
    learning = learningfunctions.Learning(costClass,paramsClass,learning_rate,l1=0.,l2=0.,maxnorm=0.,c=clippings)
    updatesClass = learning.Adam()

    print 'compiling graph you talented soul'
    classifierTrain = theano.function([X,Y], [costClass, softoutClass],
                                      updates=updatesClass, allow_input_downcast=True)
    d3v.d3viz(classifierTrain,'example.html')
    classifierPredict = theano.function([X], softoutClass, allow_input_downcast=True)
    print 'finished compiling'

    trainIndex = int(len(hexSessionsKeys)*trainPercent)

    epochCost = []
    gradNorms = []
    trainAcc = []
    testAcc = []

    costCollect = []
    trainCollect = []

    print 'training begins'
    iteration = 0
    for epoch in xrange(epochs):

        #iteration/minibatch
        for start, end in zip(range(0, trainIndex, batch_size),
                              range(batch_size, trainIndex, batch_size)):

            trainingTargets = []
            trainingSessions = []
            for trainKey in range(start, end):
                sessionForEncoding = hexSessions[hexSessions.keys()[trainKey]][0]
                #print sessionForEncoding

                adfun = adversarialfunctions.Adversary(sessionForEncoding)
                adversaryList = [sessionForEncoding,
                                 adfun.dstIpSwapOut(comsDict, uniqIPs),
                                 adfun.portDirSwitcher(),
                                 adfun.ipDirSwitcher(),
                                 adfun.noisyPacketMaker(maxPackets, packetTimeSteps, percentNoisy = 0.2)]

                if binaryTarget:
                    # choose normal and one of the abnormal types
                    abbyIndex = random.sample([0, random.sample(xrange(1,len(adversaryList)), 1)[0]], 1)[0]
                    if abbyIndex == 0:
                        targetClasses = [0,1]
                    else:
                        targetClasses = [1,0]
                else:
                    assert len(adversaryList)==numClasses
                    abbyIndex = random.sample(range(len(adversaryList)), 1)[0]
                    targetClasses = [0]*numClasses
                    targetClasses[abbyIndex] = 1

                abbyOneHotSes = oneSessionEncoder(adversaryList[abbyIndex],
                                                  hexDict = hexDict,
                                                  packetReverse=packetReverse,
                                                  padOldTimeSteps = padOldTimeSteps,
                                                  maxPackets = maxPackets,
                                                  packetTimeSteps = packetTimeSteps,
                                                  onlyEssentials = onlyEssentials)

                trainingSessions.append(abbyOneHotSes[0])
                trainingTargets.append(np.array(targetClasses, dtype=theano.config.floatX))

            sessionsMinibatch = np.asarray(trainingSessions).reshape((-1, packetTimeSteps, 1, dimIn))
            targetsMinibatch = np.asarray(trainingTargets)

            costfun = classifierTrain(sessionsMinibatch, targetsMinibatch)

            costCollect.append(costfun[0])
            trainCollect.append(np.mean(np.argmax(costfun[-1],axis=1) == np.argmax(targetsMinibatch, axis=1)))

            iteration+=1

            if iteration == 1:
                print 'you are amazing'


            # collect training stats
            if iteration%200 == 0:
                print '   Iteration: ', iteration
                print '   Cost: ', np.mean(costCollect[-20:])
                print '   TRAIN accuracy: ', np.mean(trainCollect[-20:])

                np.savetxt(savePath+runname+"_TRAIN.csv", trainCollect[::50], delimiter=",")
                np.savetxt(savePath+runname+"_COST.csv", costCollect[::50], delimiter=",")

            #testing accuracy

            if iteration%500 == 0:
                predtar, acttar, testCollect = predictClass(classifierPredict, hexSessions, comsDict, uniqIPs, hexDict,
                                                            hexSessionsKeys, binaryTarget, numClasses, onlyEssentials,
                                                            trainPercent, dimIn, maxPackets, packetTimeSteps,
                                                            padOldTimeSteps)

                binaryPrecisionRecall(predtar, acttar, numClasses)
                testAcc.append(testCollect)
                np.savetxt(savePath+runname+"_TEST.csv", testAcc, delimiter=",")

            #save the models
            if iteration%1500 == 0:
                pickleFile(classifierTrain, filePath=savePath,
                            fileName=runname+'TRAIN'+str(iteration))
                pickleFile(classifierPredict, filePath=savePath,
                            fileName=runname+'PREDICT'+str(iteration))

        epochCost.append(np.mean(costCollect[-50:]))
        trainAcc.append(np.mean(trainCollect[-50:]))

        print 'Epoch: ', epoch
        #module_logger.debug('Epoch:%r',epoch)
        print 'Epoch cost average: ', epochCost[-1]
        print 'Epoch TRAIN accuracy: ', trainAcc[-1]

    return classifierTrain, classifierPredict


if __name__=='__main__':

    maxPackets = 3
    #loadPrepedData = True  # load preprocessed data
    loadPrepedData = False  # load preprocessed data
    dataPath = '/home/andrea/Desktop/smallFlows.pcap'  # path to data
    savePath = '/home/andrea/Desktop/'  # where to save outputs
    vae = False

    packetTimeSteps = 40 # number of hex pairs
    packetReverse = False # reverse the order of packets ala seq2seq
    padOldTimeSteps = True # pad short sessions/packets at beginning(True) or end (False)
    onlyEssentials = True  # extracts only length,protocol,frag,srcIP,dstIP,srcport,dstport from header

    if onlyEssentials:
        packetTimeSteps = 16

    runname = 'beforeVAE'
    rnnType = 'gru'  # gru or lstm

    wtstd = 0.2  # standard dev for Isotropic weight initialization
    dimIn = 257  # hex has 256 characters + the <EOP> character
    dim = 100  # dimension reduction size
    clippings = 1  # for gradient clipping
    batch_size = 20
    binaryTarget = True
    if binaryTarget:
        numClasses = 2
    else:
        numClasses = 4

    epochs = 30
    lr = 0.0001
    decay = 0.9
    trainPercent = 0.9  # training testing split

    module_logger = logging.getLogger(__name__)
    train, predict = training(runname, rnnType, onlyEssentials, maxPackets, packetTimeSteps, packetReverse,
                          padOldTimeSteps, wtstd, lr, decay, clippings, dimIn, dim, binaryTarget, numClasses,
                          batch_size, epochs, trainPercent, dataPath, savePath, loadPrepedData)
