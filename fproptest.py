from fprop import Net

"""
input = [[0], [0]], [[0], [1]], [[1], [1]], [[1], [0]] #for training an XOR
output = [ [0], [1], [0], [1]]
"""

#bprop inputs
binput = [[1], [2]]
boutput = [[1]]

binput2 = [[3], [4]]
boutput2 = [[0]]

input = [[[1], [2]], [[3], [4]]]
output = [[[1]], [[0]]]
netSize = [2, 3, 1]

eta = 1.0 #learning rate

miniBatch = zip(input, output)

sadie = Net(netSize)

sadie.gradientDescent(eta, miniBatch)

#sadie.fprop(binput)

#sadie.fprop(binput2)

