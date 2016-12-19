from fprop import Net
from network import Network
import numpy as np
import pprint as pprint

#bprop inputs
input = [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]]
output = [[[1]], [[2]], [[3]], [[4]] ]

#input2 = [[3], [4]]
#boutput2 = [[0]]

netSize = [2, 3, 1]
eta = 1.0 #learning rate
miniBatch = zip(input, output)

n1 = Net(netSize)
n2 = Network(netSize)

#Set the biases and weights of our networks to equivalent values
count = 0
for x, y in zip(n1.biases, n1.weights):
	
	n2.biases[count]  = np.array(x)
	n2.weights[count] = np.array(np.transpose(y))
	

	count += 1
	
n1.gradientDescent(eta, miniBatch)

print"""
------------------------------------------------------------------------------------------------------------
NETWORK TWO
------------------------------------------------------------------------------------------------------------
"""

input = np.array(input)
output = np.array(output)
n2.update_mini_batch(zip(input, output), eta)
















#sadie.gradientDescent(eta, miniBatch)

#sadie.fprop(binput)

#sadie.fprop(binput2)

