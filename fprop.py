#Import our boy numpy
import numpy as np
import math
import random

a = [[1,2],[3,4]]
b = [[1,1,1],[1,1,1]]



"""
MAKE SURE TO OPTIMIZE YOUR ARRAY MODIFICATION CODE BY USING DEEP MODIFICATION AKA PASS BY REFERENCE
"""

def class forwardPropagation(object):

	def __init__(self):


	def array (length, width):
		"""Creates an array of zeroes with specified dimensions"""
		return [[0 for i in range(width)] for j in range(length)]

	#Use double loops to create a random array of neat stuff
	def randomArray (length, width):
		"""Creates a random array of dimensions as per length and width"""
		return [[random.random() for i in range(width)] for j in range(length)]

	def sigmoidArray(x):
		"""This simply returns the value of our sigmoid function"""
	
		numRows = len(x)
		numCols = len(x[0])
	
		return [[sigmoid(x[j][i]) for i in range(numCols)] for j in range(numRows)]
	
	def sigmoid(x):
		return (1 / (1 + math.exp(-x)))
	
	def sigmoidPrime(x):
		return math.exp(-x_/((1 + math.exp(-x))**2))
	
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
	
	def matrixAddition(a, b):
		"""Creates a matrix by adding element wise"""
	
	def forward(inputSize, hiddenSize, outputSize, hiddenCount):
		"""Initializes architecture of network"""
	
		#flag = 0
	
		#Inputs
		#For the sake of testing, hard code this value
		inputLayer = [[1,2,3],[4,5,6]]
	
		"""
	while flag == 0:
		inputLayer = raw_input("Please type inputs as a matrix")
		
		if len(inputLayer) == inputSize:
			flag = 1
		else:
			print "Invalid size. Input size is: ", inputSize
		"""
	
		#Initialize weights
		w1 = randomArray(inputLayerSize, hiddenLayerSize)
		print "w1 \n", w1
		w2 = randomArray(hiddenLayerSize, outputLayerSize)
		print "w2 \n", w2
	
		hiddenLayer = sigmoidArray(matrixMultiply(inputLayer, w1))
		print "hiddenLayer \n", hiddenLayer
		outputLayer = sigmoidArray(matrixMultiply(hiddenLayer, w2))
		print "outputLayer \n", outputLayer
	