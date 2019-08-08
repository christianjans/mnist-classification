import os
import sys
import time 

import numpy as np

import pickle

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"	# hide the pygame message
import pygame
from pygame.locals import *

from multiprocessing.pool import ThreadPool

import cjnn


class Tester:

	def __init__(self):
		# initialize variables
		self.image_size = 28
		self.image_pixels = self.image_size * self.image_size
		self.colour_factor = 1.0 / 255
		self.num_possible_vals = 10
		self.data_path = "data/mnist/"
		self.nn_path = "nn/mnist_main.p"
		self.options = ["1", "2", "3", "secret_option"]


	def test(self):
		# introduce the program
		print("Fully Connected, Deep Neural Network Test With MNIST Digits")
		print("\nThis program comes with a pre-trained neural net. Choose an option below!\n" \
			+ "- Press '1' (and then enter) to test this pre-trained neural net.\n" \
			+ "- Press '2' (and then enter) to continue to train this pre-trained neural net.\n" \
			+ "- Press '3' (and then enter) to train and test a new neural net (with desired epoch, learning rate, and hidden layers).")
		
		# get a valid choice from the user
		choice = input("Choice: ")
		while choice not in self.options:
			choice = input("That's not an option, try again: ")

		# go through the choices
		if choice == "1":
			# make sure the pre-trained neural net exists
			if os.path.isfile(self.nn_path):

				# load the pre-trained neural net
				print("Neural net exists, loading neural net...")
				with open(self.nn_path, 'rb') as f:
					self.nn = pickle.load(f)
				print("Finished loading neural net.")

				# start the pygame window to get input
				self.__start_input_window()
			else:
				# hopefully never gets here
				print("Pre-trained neural net does not exist...")
		elif choice == "2":
			# make sure the pre-trained neural net exists
			if os.path.isfile(self.nn_path):
				# initialize a threadpool to load testing data in separate thread
				pool = ThreadPool(processes = 1)

				# load the pre-trained neural net
				print("Neural net exists, loading neural net...")
				with open(self.nn_path, 'rb') as f:
					self.nn = pickle.load(f)
				print("Finished loading neural net.")

				# load training data
				(training_imgs, training_labels) = self.__get_formatted_training_data()

				# load the testing data while training the neural net
				async_result = pool.apply_async(self.__get_formatted_testing_data) 
				self.__train_neural_net(training_imgs, training_labels, 1)

				# get the testing data from the threadpool
				(testing_data, testing_labels) = async_result.get()

				# test the neural net and start the pygame input window
				self.__test_neural_net(testing_data, testing_labels)

				# save the neural net
				self.__save_nn(self.nn_path)
			else:
				# hopefully never gets here
				print("Pre-trained neural net does not exist...")
		elif choice == "3":
			# initialize a threadpool to load testing data in separate thread
			pool = ThreadPool(processes = 1)

			# get the epoch, learrning rate, and hidden layers from the user
			epoch = int(input("Enter a desired epoch: "))
			learning_rate = float(input("Enter a desired learning rate: "))
			hidden_layers = int(input("Enter the number of hidden layers: "))
			
			# get the number of neurons for each hidden layer from the user
			neurons_in_layers = []
			for i in range(hidden_layers):
				neurons_in_layers.append(int(input("Number of neurons in layer " + str(i + 1) + ": ")))

			# set the Tester's neural net to a new instance of the neural net class
			self.nn = cjnn.CJNeuralNetwork([784] + neurons_in_layers + [10], learning_rate)

			# load the training data on the main thread
			(training_data, training_labels) = self.__get_formatted_training_data()

			# load the testing data while training the neural net
			async_result = pool.apply_async(self.__get_formatted_testing_data) 
			self.__train_neural_net(training_data, training_labels, epoch)

			# get the testing data from the threadpool
			(testing_data, testing_labels) = async_result.get()

			# test the neural net and start the pygame input window
			self.__test_neural_net(testing_data, testing_labels)
			self.__start_input_window()
		elif choice == "secret_option":
			# this case is just used to train a new main neural net.
			# should probably not use this case unless you want to 
			# replace the main neural net with one that will have
			# done 10 epochs (and may be less accurate) 

			# initalize some variables
			epoch = 10
			learning_rate = 0.1
			pool = ThreadPool(processes = 1)
			self.nn = cjnn.CJNeuralNetwork([784, 32, 32, 10], learning_rate)

			# load training & testing data
			(training_data, training_labels) = self.__get_formatted_training_data()
			async_result = pool.apply_async(self.__get_formatted_testing_data)
			
			# train neural net
			self.__train_neural_net(training_data, training_labels, epoch)

			# get the testing data
			(testing_data, testing_labels) = async_result.get()

			# test, save, and use the input window for testing
			self.__test_neural_net(testing_data, testing_labels)
			self.__save_nn(self.nn_path)
			self.__start_input_window()

	# train the Tester's neural net with training data, training labels, and epoch
	def __train_neural_net(self, training_data, training_labels, epoch):
		print("Training neural net...")

		initial_time = int(round(time.time() * 1000))

		# loop through each epoch
		for j in range(0, epoch):
			# loop through each element in the training data
			for i in range(0, len(training_data)):
				# train the neural net
				self.nn.feed_forward(training_data[i])
				self.nn.backpropagate(self.__label_to_arr(training_labels[i]))

				# print progress every 1000 milliseconds
				current_time = int(round(time.time() * 1000))
				if current_time - initial_time > 1000:
					print("Completed " + str(i + 1) + " of " + str(len(training_data)) + " training examples")
					initial_time = current_time
			print("\nFinished epoch " + str(j + 1) + " of " + str(epoch) + "\n")


	# test the Tester's neural net with testing data, testing labels, and an option to display output
	def __test_neural_net(self, testing_data, testing_labels, display_output = False):
		print("Testing neural net...")

		num_correct = 0;

		# loop through each element in the testing data
		for i in range(0, len(testing_data)):
			# get the neural net's output
			guess = self.nn.feed_forward(testing_data[i])

			# one-hot encode the guess and see if it matches the answer
			if self.__arr_to_label(guess) == testing_labels[i]:
				num_correct = num_correct + 1

			# print every 100th guess if display_output is True
			if i % 100 == 0 and display_output == True:
				print("\nNumber is:", int(testing_labels[i]))

				percentages = 100 * guess
				print("Net guess:")
				for i in range(0, len(percentages)):
					histogram = ""
					for j in range(0, int(100 * guess[i] / 2)):
						histogram += u"\u2588"
					print("Probability that it's a " + str(i) + ": " + histogram + " (%.2f)" % round(100 * float(guess[i]), 2) + "%")

		print("Got " + str(num_correct) + "/" + str(len(testing_labels)))


	# get training data, formatted so that it has an input and an answer
	def __get_formatted_training_data(self):
		training_imgs = []

		print("Loading training dataset... (this may take a minute or two)")
		training_imgs = np.loadtxt(self.data_path + "mnist_train.csv", delimiter = ",")
		print("Finished loading training dataset.")

		print("Creating training images...")
		training_data = np.asfarray(training_imgs[:, 1:]) * self.colour_factor
		print("Finished creating training images.")

		print("Creating training labels...")
		training_labels = np.asfarray(training_imgs[:, :1])
		print("Finished creating training labels.")

		return (training_data, training_labels)


	# get testing data, formatted so that it has an input and an answer
	def __get_formatted_testing_data(self):
		testing_imgs = []

		print("Loading testing dataset...")
		testing_imgs = np.loadtxt(self.data_path + "mnist_test.csv", delimiter = ",")
		print("Finished loading testing dataset.")

		print("Creating testing images...")
		testing_data = np.asfarray(testing_imgs[:, 1:]) * self.colour_factor
		print("Finished creating testing images.")

		print("Creating testing labels...")
		testing_labels = np.asfarray(testing_imgs[:, :1])
		print("Finished creating testing labels.")

		return (testing_data, testing_labels)


	# save the main neural net using pickle
	def __save_nn(self, path):
		print("Saving neural net...")
		with open(path, 'wb') as f:
			pickle.dump(self.nn, f, protocol = pickle.HIGHEST_PROTOCOL)
		print("Finished saving neural net.")


	# start a pygame window that can take in a drawn input
	def __start_input_window(self):
		pygame.init()
		mouse = pygame.mouse

		width = 280
		height = 280
		WHITE = pygame.Color(0, 0, 0)
		BLACK = pygame.Color(255, 255, 255)

		window = pygame.display.set_mode((width, height))
		canvas = window.copy()

		canvas.fill(WHITE)

		# give instructions
		print("\nDraw on the canvas to test the neural net's digit recognition (0 - 9).")
		print("- Press SPACE to clear drawing.")
		print("- Press ENTER to submit drawing for testing.")

		# main 'game' loop
		while True:
			left_pressed, middle_pressed, right_pressed = mouse.get_pressed()
			keys = pygame.key.get_pressed()

			for event in pygame.event.get():
				if event.type == QUIT:	# exit the window
					pygame.quit()
					sys.exit()
				elif left_pressed:		# draw a circle
					pygame.draw.circle(canvas, BLACK, (pygame.mouse.get_pos()), 12)
				elif keys[K_SPACE]:		# reset the canvas
					canvas.fill(WHITE)
				elif keys[K_RETURN]:	# convert the canvas to an input array for the neural net
					pixels = []

					for j in range(0, height, 10):
						for i in range(0, width, 10):
							pixel = canvas.get_at((i, j))
							colour = (pixel.r + pixel.g + pixel.b) / (3 * 255.0)
							pixels.append(colour)

					# get the neural net's guess
					guess = self.nn.feed_forward(pixels)

					# print the guess of the neural net (using histogram)
					print("\nNet's guess is " + str(self.__arr_to_label(guess)))
					for i in range(0, len(guess)):
						histogram = ""
						for j in range(0, int(100 * guess[i] / 2)):
							histogram += u"\u2588"
						print("Probability that it's a " + str(i) + ": " + histogram + " (%.2f)" % round(100 * float(guess[i]), 2) + "%")

			# refresh the window with the canvas
			window.fill(WHITE)
			window.blit(canvas, (0, 0))
			pygame.display.update()


	# converts a number (0 - 9) to a one-hot encoded vector
	def __label_to_arr(self, label):
		arr = np.zeros((1, 10))
		arr[0, int(label)] = 1
		return arr


	# converts a one-hot encoded vector to a number (0 - 9)
	def __arr_to_label(self, arr):
		l = arr.tolist()
		m = l[0][0]

		for i in range(1, len(l)):
			if l[i][0] > m:
				m = l[i][0]

		return l.index([m])


# start and test a new instance of the Tester class
if __name__ == "__main__":
	tester = Tester()
	tester.test()


