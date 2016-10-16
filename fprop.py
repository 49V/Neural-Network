import math
import random
from pprint import pprint

class NeuralNetwork(object):

	def init(input, netSize):
		"""Initializes architecture of network"""
	
		#Inputs
		#For the sake of testing, hard code this value
		a = [[1], [2]]
	
		#Initialize Weights
		#This initializes the weights for each layer based on the size. The number of rows should be
		#the number of neurons for the current, and the number of columns should be the same as the number of neurons
		#in the next layer
		weights = [randomArray(x, y) for x, y in zip(netSize[:-1], netSize[1:]) ] # netSize[:-1] continues until the 2nd last element in the list
	
		#Initialize Biases
		#Similar sort of thing for biases. We don't have a bias for the first, input layer
		biases = [randomArray(x, 1) for x in netSize[1:]] 
	
		#First create network based on size


	
	def fprop(input, netSize):
		"""Fprop takes an input and propagates it through the network"""
	
		weights = [randomArray(x, y) for x, y in zip(netSize[:-1], netSize[1:]) ]
		biases = [randomArray(x, 1) for x in netSize[1:]]
	
		#Initialize a, and z as lists. Remember, we need z to compute the error!
		z = [array(i, 1) for i in netSize]
		a = [array(i, 1) for i in netSize]
	
		#Assign the input to first layer of activations
		z[0] = input
		a[0] = input
	
		runs = len(weights[0])
	
		for i in range(runs):
			z[i + 1] = matrixMultiply(transpose(weights[i]),z[i]) 
		
		#Remember, we don't computer sigmoid of our input layer
		for i in (range(1, len(z))):
			a[i] = sigmoidArray(z[i])
		
		return a, z, weights, biases, netSize
	#----------------------------------------------------------------------------------------------------------    

	#For test purposes just use netSize
	def bprop(a, z, weights, biases,netSize):
		"""
		Compute the gradients for the biases and the weights for a given input and ouput tuple (x, y)
	
		a       : Matrix of Activations
		z       : Matrix of Weighted Inputs
		weights : Matrix containing all weights
		biases  : Matrix containing all biases
		netSize : List containing sizing dimensions of a network i.e. (2, 3, 1) means a networks that
		has 3 layers. 2 neurons in the input layer, 3 neurons in the hidden layer, and 1 neuron in the 
		output layer.
		"""

		"""
		1) Initialization
		"""
		outputDesired = [[0.5]]
		#Initialize delta
		delta = [array(i, 1) for i in netSize]
		print "Delta:"
		pprint(delta)
	
		"""
		2) Compute output error, delta
		"""
		delta[-1] = hadmardProduct((matrixSubtraction(a[-1], outputDesired)), sigmoidPrimeArray(z[-1]))
		print "Delta - 1"
		pprint(delta)
	
		#We want to go from the second last value until the second value!
		interval = range(len(netSize) - 2, 0, -1)
	
		"""
		3) Backpropagate the error to all previous layers.
		"""
		#Now it's time to backpropagate
		#Remember, only weights for in between each layer!
		delta[i] = [hadmardProduct(matrixMultiply(weights[i], delta[i + 1]), sigmoidPrimeArray(z[i])) for i in interval]
	
		"""
		4) Compute Gradients.
		"""
		#We have weights for every layer except for the last
		numLayers = len(netSize[0])
		gradient_weights[i] =[matrixMultiply(delta[i], transpose(a[-i - 1])) for i in range(2, numLayers)] 
		#We have biases for every layer but the first, and the last layer
		gradient_biases = delta
	
	
		print "\n DELTA FINAL: \n"
		pprint(delta)
	
		print "\n GRADIENT WEIGHTS: \n"
		pprint(gradient_weights)
	
		print "\n GRADIENT BIASES: \n"
		pprint(gradient_biases)
	
	
		return gradient_biases, gradient_weights

	def gradientDescent(eta, miniBatch, netSize):
		"""
		#Takes a learning rate eta, and a mini batch (miniBatch) and returns updated biases and weights
		#based on gradients.
		"""
		batchSize = len(miniBatch)
	
		#REALLY INEFFICIENT, NEED TO USE LESS LISTS! SO MANY RESOURCES MOSHE! Alright for POC
		sum_biases  = [array(i,1) for i in netSize[1:]]
		temp_biases = [array(i,1) for i in netSize[1:]]
		eta_biases  = [array(i,1) for i in netSize[1:]] 
	
		sum_weights  = [[array(i,j) for i,j in zip(netSize[:-1], netSize[1:])]]
		temp_weights = [[array(i,j) for i,j in zip(netSize[:-1], netSize[1:])]]
		eta_weights  = [[array(i,j) for i,j in zip(netSize[:-1], netSize[1:])]]
	
		#First calculate the sum of the gradients of the biases, and weights respectively
		for x,y in miniBatch:
			temp_biases, temp_weights = bprop(x, y, netSize)
			#Remember you have lists within lists, can't just add.
			sum_biases  = [matrixAddition(i, j) for i, j in zip(sum_biases, temp_biases)]
			sum_weights = [matrixAddition(i, j) for i, j in zip(sum_weights, temp_weights)]

		#Then update the biases and weights using the learning rate and batch size
		biases  = [matrixSubtraction(b, ((eta/batchSize) * sb)) for b, sb in zip(biases, sum_biases)]
		weights = [matrixSubtraction(w, ((eta/batchSize) * sw)) for w, sw in zip(weights, sum_weights)]
	
		return biases, weights
	
	def array (length, width):
		"""Creates an array of zeroes with specified dimensions"""
		return [[0 for i in range(width)] for j in range(length)] #List comprehensions (Works like two for loops)

	def matrixElementMultiply (value, list):
		"""Multiplies each element of a list by the given value"""
		numRows = len(list)
		numCols = len(list[0])
	
		return [[value * list[j][i] for i in range(numCols)] for j in range(numRows)]
	
	#Use double loops to create a random array of neat stuff
	def randomArray (length, width):
		"""Creates a random array of dimensions as per length and width"""
		return [[random.random() for 
		i in range(width)] for j in range(length)]

	def sigmoidArray(x):
		"""This simply returns the value of our sigmoid function"""
	
		numRows = len(x)
		numCols = len(x[0])
	
		return [[sigmoid(x[j][i]) for i in range(numCols)] for j in range(numRows)]
	
	def sigmoid(x):
		return (1 / (1 + math.exp(-x)))
	
	def sigmoidPrime(x):
		return math.exp(x)/((1 + math.exp(-x))**2)
	
	def sigmoidPrimeArray(x):
		numRows = len(x)
		numCols = len(x[0])
	
		return [[sigmoidPrime(x[j][i]) for i in range(numCols)] for j in range(numRows)]
	
	def matrixMultiply(a, b):

		#Number of rows of matrix a
		numRowsA = len(a)
		#Number of rows and columns of matrix b
		numRowsB = len(b)
		numColsB = len(b[0])
	
		c = array(numRowsA, numColsB)
	
		#Loop through rows of first matrix
		for i in range(numRowsA):
		#Loop through columns of your second
			for j in range (numColsB):
				for k in range (numRowsB):
					c[i][j] += a[i][k] * b[k][j]
			
		return c
	

	def dotProduct(a, b):
		length = len(a)
		width  = len(a[0])
		return [[a[i][j] * b[i][j] for i in range(width)] for j in range(length)]

	def dotProductArray(a, b):
		length = len(a)
		width  = len(a[0])
	
		return [[dotProduct(a[i], b[i]) for i in range(length)] for j in range(width)]
	
	def matrixAddition(a, b):
		"""Creates a matrix by adding element wise. Matrices must have matching dimensions"""
	
		numRows = len(a)
		numCols = len(a[0])
	
		return [[a[j][i] + b[j][i] for i in range(numCols)] for j in range(numRows)]
	
	def matrixSubtraction(a, b):
		"""Creates a matrix by subtracting elementwise. Matrices must have matching dimensions"""
	
		numRows = len(a)
		numCols = len(a[0])
	
		return [[ a[j][i] - b[j][i] for i in range(numCols)] for j in range(numRows)]
	
	def hadmardProduct(a,b):
		"""Computes hadmard product of two matrices. Matrices must have matching dimensions"""
		numRows = len(a)
		numCols = len(a[0])
	
		return [[a[j][i] * b[j][i] for i in range(numCols)] for j in range(numRows)]
	
	def transpose(a):
		numRows = len(a)
		numCols = len(a[0])
	
		b = array(numCols, numRows)
	
		for i in range(numRows):
			for j in range(numCols):
				b[j][i] = a[i][j] 
		return b
	
	
