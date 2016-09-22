#Import our boy numpy
import numpy as np
import math
import random
from pprint import pprint

a = [[1,2],[3,4]]
b = [[1,1,1],[1,1,1]]

inputLayerSize  = 2
hiddenLayerSize = 3
outputLayerSize = 1

"""
MAKE SURE TO OPTIMIZE YOUR ARRAY MODIFICATION CODE BY USING DEEP MODIFICATION AKA PASS BY REFERENCE
"""

def array (length, width):
	"""Creates an array of zeroes with specified dimensions"""
	return [[0 for i in range(width)] for j in range(length)] #List comprehensions (Works like two for loops)

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

#def dotProductArray(a, b):
	#length = len(a)
	#width  = len(a[0])
	
	#return [[dotProduct(a[i], b[i]) for i in the range
	
def matrixAddition(a, b):
	"""Creates a matrix by adding element wise. Matrices must have matching dimensions"""
	
	numRows = len(a)
	numCols = len(a[0])
	
	return [[a[j][i] + b[j][i] for i in range(numCols)] for j in range(numRows)]
	
def hadmardProduct(a,b):
	"""Computes hadmard product of two matrices. Matrices must have matching dimensions"""
	numRows = len(a)
	numCols = len(a[0])
	
	return [[a[j][i] * b[j][i] for i in range(numCols)] for j in range(numRows)]
	
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
	

def fprop(x, netSize):
	"""Fprop takes an input and propagates it through the network"""
	weights = [randomArray(x, y) for x, y in zip(netSize[:-1], netSize[1:]) ]
	print "Weights \n "
	pprint(weights)
	biases = [randomArray(x, 1) for x in netSize[1:]]
	print "\n Biases \n"
	pprint(biases)
	
	#Initialize a as a list
	a = [array(x, 1) for x in netSize]
	print" a:"
	pprint(a)
	#Assign the input to first layer of activations
	a[0] = x
	
	runs = len(weights[0])
	print "Runs, ", runs
	
	for i in range(runs):
		a[i + 1] = matrixMultiply(a[i],weights[i]) #
	
	pprint(a)	
	return a
#----------------------------------------------------------------------------------------------------------    

	
	