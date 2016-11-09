from fprop import Net

"""
input = [[0], [0]], [[0], [1]], [[1], [1]], [[1], [0]] #for training an XOR
output = [ [0], [1], [0], [1]]
"""

#bprop inputs
binput = [[1], [2]]
boutput = [[1]]

input = [[[1], [2]]]
output = [[[1]]]
netSize = [2, 3, 1]

eta = 1 #learning rate

miniBatch = zip(input, output)

sadie = Net(netSize)

sadie.bprop(binput, boutput)

print """ BACKPROP IS FINISHED
----------------------------
----------------------------
----------------------------
----------------------------
----------------------------
----------------------------
----------------------------
"""

sadie.gradientDescent(eta, miniBatch)