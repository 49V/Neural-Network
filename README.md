# Neural-Network
I am currently working with a group of students acting as the president for a design team at the University of British Columbia. We are
researching the applications of neural networks and how they might be implemented using FPGAs (Field-Programmable Gate Arrays). We are doing
this in two main steps. The first step involves developing a rigorous understanding of Convolutional Neural Networks by first implementing
them in software. The current language of choice is Python 2.7. The second step is to implement the network using the hardware design 
language VHDL, which will come much later.

Currently this repository will follows the steps of the creation for every single function needed for a functional neural network. We are
first starting by implementing neural networks without deep learning i.e. only a single hidden layer. As we progress we will improve our
technique and add complexity to our code. The current reference material being used is http://neuralnetworksanddeeplearning.com/ , an online
textbook by Michael Nielsen. We cluster the learning into groups of one or two chapters and first start by learning the theory behind neural
networks. After finishing reading we follow up by writing the code ourselves. We do not use the code in the textbook (aside from playing
around with basic concepts). All of the code written is mostly original and only uses the textbook as reference material.

VHDL is a much more strongly typed language than Python and ultimately lacks a lot of the basic functions that most Python users take for 
granted. We believe that by writing code for neural networks, include all of the math functions, will allow us to develop similar methods
for implementation in VHDL. There is no direct translation from hardware to software and we are eagerly looking forward to using FPGAs and 
comparing differences in processing time and power consumption.

By splitting the learning into equal chunks of basic and applied knowledge it helps with the mental digestion of neural networks and more
importantly it helps the team keep on task. If you're interested in learning more about neural networks we would love to have you join the 
team. If you already have a strong knowledge, mentoriship opportunities are also a possibility. Our budget varies on a yearly basis, but we
are always willing to seek more funding as our team grows.

Check out the design team at http://ubchdt.com/.
