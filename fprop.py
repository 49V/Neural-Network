import math
import random
from pprint import pprint


class Net(object):
	"""
	Neural Network classes that creates a neural network objects.
	
	z(l) = (a(l-1) * w(l-1)) + b
	a(l) = z(l)
	Attributes:
		activations			: Activations of the neural network are sigmoid(z). Activations are output levels
							  for individual neurons. If an activation level exceeds a threshold, a neuron will fire.
		biases				: The biases are just the threshold level of an individual neuron that must be exceeded
							  in order to "fire" a neuron.
		gradient_biases		: The gradient of the cost function with respect to the biases.
		gradient_weights	: The gradient of the cost function with respect to the weights.
		input*				: Inputs to a neural network.
		netSize*			: The size of each layer of a neural network. For example a netSize of [2, 3, 1]
							  corresponds to 3 layer network with 2 input neurons, 3 hidden neurons, and 1 output	
							  neuron.
		outputs				: The output of a neural network.
		weights				: The weights of a neural network determine how large of a value an input neuron
							  contributes for a fixed activation. For example two neurons could each have activation of 0.5, 0.5
							  respectively. If they had weights of [1, 2] respectively, the second neuron would have twice as much 
							  of an effect as an input. This neuron is more weighted.
		z				    : The weighted input. This is simply the activation without a sigmoid function applied : (a(l-1) * w(l-1)) + b 
	"""
	def __init__(self, netSize):
		"""Initializes all the attributes of a Neural Network object"""
		
		# TRY THIS FOR RANDOM!
		#
		#
		#
		
		self.biases           = [self.randomArray(i, 1) for i in netSize[1:]]  # Biases do not exist for the first layer ! Those are inputs.
		self.netSize          = netSize
		#Initialize Weights
		#This initializes the weights for each layer based on the size. The number of rows should be
		#the number of neurons for the current, and the number of columns should be the same as the number of neurons
		#in the next layer. There are no weights for the last layer. That's the output layer.
		self.weights 		  = [self.randomArray(i, j) for i, j in zip(netSize[:-1], netSize[1:]) ] 
		
	def fprop(self, input):
		"""
		fprop takes an input and propagates it through the network.
		z(l) = (a(l-1) * w(l-1)) + b(l)
		a(l) = z(l)
		
		Returns activations, and weighted inputs
		"""
		activations      = [self.array(i, 1) for i in self.netSize]		# Activations have same dimensions as network	
		z                = [self.array(i, 1) for i in self.netSize]      # Weighted input have same dimensions as network
	
		
		#Assign the input to first layer of activations, and weighted inputs
		z[0] = input
		activations[0] = input
		runs = len(self.weights[0])
		
		#Propagate z through the network.
		for i in range(runs):
			z[i + 1] = self.matrixAddition(self.matrixMultiply(self.transpose(self.weights[i]), activations[i]), self.biases[i]) 
			activations[i + 1] = self.sigmoidArray(z[i + 1])
			
			
		print "OUTPUTS:"	
		pprint (activations)
		return activations, z
	
	def bprop(self, input, output):
		"""
		Compute the gradients for the biases and the weights for a given input and ouput tuple (x, y)
		"""

		gradient_biases  = [self.array(i, 1) for i in self.netSize[1:]]
		gradient_weights = [self.array(i, j) for i, j in zip(self.netSize[:-1], self.netSize[1:])]
		
		"""
		First we should obtain our activations and z for a given input
		"""
		a, z = self.fprop(input)
		
		print "a: "
		pprint(a)
		print "z: "
		pprint(z)
		
		"""
		1) Initialization
		"""
		
		#Initialize delta
		delta = [self.array(i, 1) for i in self.netSize[1:]]
				
		"""
		2) Compute output error, delta
		"""
		delta[-1] = self.hadmardProduct((self.matrixSubtraction(a[-1], output)), self.sigmoidPrimeArray(z[-1]))
		print "DELTA -1"
		print delta[-1]


		#We want to go from the second last value until the second value!
		interval = range(len(self.netSize) - 2, 0, -1)
	
		"""
		3) Backpropagate the error to all previous layers.
		"""
		#Now it's time to backpropagate
		#Remember, only weights for in between each layer!
		#BE VERY CAREFUL WHEN USING LIST COMPREHENSIONS!
		for i in interval:
			delta[i - 1] = self.hadmardProduct(self.matrixMultiply(self.weights[i], delta[i]), self.sigmoidPrimeArray(z[i])) 
	
		"""
		4) Compute Gradients.
		"""
		#We have weights for every layer except for the last
		numLayers = len(self.netSize)
		
		"""
		Remember how you arrange weights.
		You arrange weights from the jth neuron in the lth layer, to the kth neuron in the lth + 1 layer
		     -
		-   
		     -
		-
		     -
		Because of the way you have it arranged, the weights only exist for every layer except for the last!	 
		So for the gradient with respect to the weight, the formula is d/dw (l) = a(l) * delta(i + 1)
		It isn't the same as in the book!
		
		This should also be activations * delta ^T (delta transpose)
		"""
		for i in range(0, numLayers - 1):
			gradient_weights[i] =self.matrixMultiply(a[i], self.transpose(delta[i]))  
		
		#We have biases for every layer but the first, and the last layer
		gradient_biases = delta
	
		print "\n DELTA FINAL: \n"
		pprint(delta)
		print "\n GRADIENT WEIGHTS: \n"
		pprint(gradient_weights)
		print "\n GRADIENT BIASES: \n"
		pprint(gradient_biases)
	
		return gradient_biases, gradient_weights

	def gradientDescent(self, eta, miniBatch):
		"""
		#Takes a learning rate eta, and a mini batch (miniBatch) and updates biases and weights
		#based on gradients. A mini batch is just a subset of the input and output tuples (x, y)
		"""
		
		print "Minibatch: "
		pprint(miniBatch)
		
		print "Weights"
		pprint(self.weights)
		
		print "Biases"
		pprint(self.biases)

		
		#assuming input/output tuples
		batchSize = len(miniBatch)
	
		#Create matrices to hold the sums of the gradients for a given minibatch
		sum_biases  = [self.array(i, 1) for i in self.netSize[1:]]
		sum_weights  = [self.array(i, j) for i, j in zip(self.netSize[:-1], self.netSize[1:])]
	
		#First calculate the sum of the gradients of the biases, and weights respectively
		for i, j in miniBatch:
		
			temp_biases, temp_weights = self.bprop(i, j)
			
			print """
				  PASS---------------------------------------------------------------------------- 
				  """
			
			print "I"
			pprint (i)
			
			print "J"
			pprint(j)
			
			print " sum biases: "
			pprint(sum_biases)
			print " temp biases: "
			pprint(temp_biases)
			
			#Remember you have lists within lists, can't just add.
			sum_biases  = [self.matrixAddition(k, l) for k, l in zip(sum_biases, temp_biases)]
			sum_weights = [self.matrixAddition(k, l) for k, l in zip(sum_weights, temp_weights)]

		print "GRADIENT BIASES (GRADIENT DESCENT): \n"
		pprint(sum_biases)
		
		print "GRADIENT WEIGHTS (GRADIENT DESCENT): \n"
		pprint(sum_weights)
		
		#Then update the biases and weights using the learning rate and batch size
		
		#MAKE SURE TO TURN ETA INTO A FLOAT OR ELSE YOU WILL GET INTEGER MULTIPLICATION!
		self.biases  = [self.matrixSubtraction(b, self.matrixElementMultiply((eta/batchSize), sb)) for b, sb in zip(self.biases, sum_biases)]
		
		"""
		for b, sb in zip(self.biases, sum_biases):
			print "B"
			pprint(b)
			
			print "SB"
			pprint(sb)
			
			print "MATH1"
			math1 = self.matrixElementMultiply((eta/batchSize), sb)
			pprint(math1)
			
			print "MATH"
			math = self.matrixSubtraction(b, math1)
			pprint(math)
		"""
			
		self.weights = [self.matrixSubtraction(w, self.matrixElementMultiply((eta/batchSize), sw)) for w, sw in zip(self.weights, sum_weights)]
		
		print "FINAL BIASES"
		pprint(self.biases)
		
		print "FINAL WEIGHTS"
		pprint(self.weights)
	
	def array (self, length, width):
		"""Creates an array of zeroes with specified dimensions"""
		return [[0 for i in range(width)] for j in range(length)] #List comprehensions (Works like two for loops)
	
	
	def matrixElementMultiply (self, value, list):
		"""Multiplies each element of a list by the given value"""
		numRows = len(list)
		numCols = len(list[0])
	
		return [[value * list[j][i] for i in range(numCols)] for j in range(numRows)]
	
	
	def randomArray (self, length, width):
		"""Creates a random array of dimensions as per length and width"""
		return [[random.random() for 
		i in range(width)] for j in range(length)]
	
	
	def sigmoidArray(self, x):
		"""This simply returns the value of our sigmoid function"""
	
		numRows = len(x)
		numCols = len(x[0])
	
		return [[self.sigmoid(x[j][i]) for i in range(numCols)] for j in range(numRows)]
	
	
	def sigmoid(self, x):
		return (1.0 / (1.0 + math.exp(-x)))
		
	
	def sigmoidPrime(self, x):
		return (self.sigmoid(x) * (1 - self.sigmoid(x))) 
		
	
	def sigmoidPrimeArray(self, x):
		numRows = len(x)
		numCols = len(x[0])
	
		return [[self.sigmoidPrime(x[j][i]) for i in range(numCols)] for j in range(numRows)]

	
	def matrixMultiply(self, a, b):

		#Number of rows of matrix a
		numRowsA = len(a)
		#Number of rows and columns of matrix b
		numRowsB = len(b)
		numColsB = len(b[0])
	
		c = self.array(numRowsA, numColsB)
	
		#Loop through rows of first matrix
		for i in range(numRowsA):
		#Loop through columns of your second
			for j in range (numColsB):
				for k in range (numRowsB):
					c[i][j] += a[i][k] * b[k][j]
			
		return c
	
	
	def dotProduct(self, a, b):
		length = len(a)
		width  = len(a[0])
		return [[a[i][j] * b[i][j] for i in range(width)] for j in range(length)]

	
	def dotProductArray(self, a, b):
		length = len(a)
		width  = len(a[0])
	
		return [[self.dotProduct(a[i], b[i]) for i in range(length)] for j in range(width)]

	
	def matrixAddition(self, a, b):
		"""Creates a matrix by adding element wise. Matrices must have matching dimensions"""
	
		numRows = len(a)
		numCols = len(a[0])
	
		return [[a[j][i] + b[j][i] for i in range(numCols)] for j in range(numRows)]

	
	def matrixSubtraction(self, a, b):
		"""Creates a matrix by subtracting elementwise. Matrices must have matching dimensions"""
	
		numRows = len(a)
		numCols = len(a[0])
	
		return [[ a[j][i] - b[j][i] for i in range(numCols)] for j in range(numRows)]

	
	def hadmardProduct(self, a,b):
		"""Computes hadmard product of two matrices. Matrices must have matching dimensions"""
		numRows = len(a)
		numCols = len(a[0])
	
		return [[a[j][i] * b[j][i] for i in range(numCols)] for j in range(numRows)]

	
	def transpose(self, a):
		numRows = len(a)
		numCols = len(a[0])
	
		b = self.array(numCols, numRows)
	
		for i in range(numRows):
			for j in range(numCols):
				b[j][i] = a[i][j] 
		return b
	
	
