import os
import math
import sys
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pylab import *
from pybrain.structure import *

ds = SupervisedDataSet(2,1)

dataset_input_file = open("hw1data.dat","rU")
dataset_origin = dataset_input_file.read()
dataset_origin = dataset_origin.split('\n')
if len(dataset_origin[len(dataset_origin)-1]) == 0:
    dataset_origin = dataset_origin[:-1]
print len(dataset_origin)
for i in range(len(dataset_origin)):
    dataset_origin[i] = dataset_origin[i].split('\t')
dataset_input_file.close()
for i in range(len(dataset_origin)):
    ds.addSample([float(dataset_origin[i][0]),float(dataset_origin[i][1])], (int(dataset_origin[i][2]),))

net = FeedForwardNetwork()
inLayer = LinearLayer(2, name='inLayer')
hiddenLayer0 = SigmoidLayer(4, name='hiddenLayer0')
hiddenLayer1 = SigmoidLayer(3, name='hiddenLayer1')
outLayer = LinearLayer(1, name='outLayer')

#build up neural network

net.addInputModule(inLayer)
net.addModule(hiddenLayer0)
net.addModule(hiddenLayer1)
net.addOutputModule(outLayer)

in_to_hidden0 = FullConnection(inLayer, hiddenLayer0)
hidden0_to_hidden1 = FullConnection(hiddenLayer0, hiddenLayer1)
hidden1_to_out = FullConnection(hiddenLayer1, outLayer)

net.addConnection(in_to_hidden0)
net.addConnection(hidden0_to_hidden1)
net.addConnection(hidden1_to_out)

net.sortModules()

#net = buildNetwork(2, 4, 3, 1, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds, verbose = True, learningrate=0.01)
#trainer = BackpropTrainer(net, ds)

net_output_file = open("hw1data.net",'w')

print net
print >> net_output_file, net

for mod in net.modules:
    print "Module:", mod.name
    print >> net_output_file,"Module:", mod.name
    if mod.paramdim > 0:
        print "--parameters:", mod.params
        print >> net_output_file, "--parameters:", mod.params
    for conn in net.connections[mod]:
        print >> net_output_file, "-connection to", conn.outmod.name
        print "-connection to", conn.outmod.name
        if conn.paramdim > 0:
            print >> net_output_file, "- parameters", conn.params
            print "- parameters", conn.params
    if hasattr(net, "recurrentConns"):
        print "Recurrent connections"
    for conn in net.connections[mod]:
        print >> net_output_file , "-", conn.inmod.name, " to", conn.outmod.name
        print "-", conn.inmod.name, " to", conn.outmod.name
        if conn.paramdim > 0:
            print >> net_output_file, "- parameters", conn.params
            print "- parameters", conn.params

trainer_message = trainer.trainUntilConvergence(maxEpochs=1000)

print >> net_output_file, "train-errors:"
net_output_file.write('[\t')
for i in range(len(trainer_message[0])):
    net_output_file.write(str(trainer_message[0][i]))
    if i!=len(trainer_message[0])-1:
        net_output_file.write('\t')
net_output_file.write(']\n')

print >> net_output_file, "valid-errors:"
net_output_file.write('[\t')
for i in range(len(trainer_message[1])):
    net_output_file.write(str(trainer_message[1][i]))
    if i!=len(trainer_message[1])-1:
        net_output_file.write('\t')
net_output_file.write(']\n')

net_output_file.close()
