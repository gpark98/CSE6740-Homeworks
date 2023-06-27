from array import array
import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os
import time

def mykmeans(pixels, K):
	pixels = pixels.reshape(pixels.shape[0] * pixels.shape[1], pixels.shape[2])
	N = pixels.shape[0]
	classes = np.zeros((N,1))
	centers = 255 * np.random.rand(K,3) # random initialization
	centers_new = np.zeros((K,3))
	distance = np.zeros((K,1))
	diff = 1
	while(diff != 0):
		# assignment
		for i in range(N):
			for j in range(K):
				distance[j] = np.linalg.norm(pixels[i] - centers[j])
			classes[i] = np.argmin(distance)
		# adjustment
		for i in range(K):
			tmp = np.zeros(3,)
			size_cluster = 0
			for j in range(N):
				if classes[j] == i:
					tmp += pixels[j]
					size_cluster += 1
			if size_cluster > 0:
				tmp /= size_cluster
			centers_new[i] = tmp
		diff = np.linalg.norm(centers - centers_new)
		centers = centers_new
		
	centers = np.around(centers)

	return classes, centers


def mykmedoids(pixels, K):

	pixels = pixels.reshape(pixels.shape[0] * pixels.shape[1], pixels.shape[2])
	N = pixels.shape[0]
	classes = np.zeros((N,1))

	from numpy.random import default_rng
	rng = default_rng()
	numbers = rng.choice(N, size=K, replace=False)

	centers = np.zeros((K,3))
	for i in range(K):
		centers[i] = pixels[int(numbers[i])]

	centers_new = np.zeros((K,3))
	distance = np.zeros((K,1))
	diff = 1
	while(diff != 0):
		# assignment
		for i in range(N):
			for j in range(K):
				distance[j] = np.linalg.norm(pixels[i] - centers[j], np.inf)
			classes[i] = np.argmin(distance)
		# adjustment
		for i in range(K):
			tmp = []
			for j in range(N):
				if classes[j] == i:
					tmp.append(j)
			if len(tmp) == 0:
				print("Warning")
			
			tmp_rep = np.zeros((1,3))
			for j in range(len(tmp)):
				tmp_rep += pixels[tmp[j]]
			tmp_rep /= len(tmp)

			distance_inside = np.zeros(len(tmp))
			for j in range(len(tmp)):
				distance_inside[j] = np.linalg.norm(pixels[tmp[j]] - tmp_rep, np.inf)

			centers_new[i] = pixels[tmp[np.argmin(distance_inside)]]
		
		diff = np.linalg.norm(centers - centers_new)
		centers = centers_new

	return classes, centers



	
def main():
	if(len(sys.argv) < 2):
		print("Please supply an image file")
		return

	image_file_name = sys.argv[1]
	K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
	print(image_file_name, K)
	im = np.asarray(imageio.imread(image_file_name))

	fig, axs = plt.subplots(1, 2)

	# start_time = time.time()

	classes, centers = mykmedoids(im, K)
	# print(classes, centers)

	# I created new_array because centers[classes] causes error
	new_array = np.zeros((classes.shape[0], 3))
	for i in range(len(new_array)):
		new_array[i] = centers[int(classes[i])]

	new_im = np.asarray(new_array.reshape(im.shape), im.dtype)
	# new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[0].imshow(new_im)
	axs[0].set_title('K-medoids')

	

	classes, centers = mykmeans(im, K)
	# print(classes, centers)

	# I created new_array because centers[classes] causes error
	new_array = np.zeros((classes.shape[0], 3))
	for i in range(len(new_array)):
		new_array[i] = centers[int(classes[i])]

	new_im = np.asarray(new_array.reshape(im.shape), im.dtype)
	# new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[1].imshow(new_im)
	axs[1].set_title('K-means')

	# print("--- %s seconds ---" % (time.time() - start_time))

	plt.show()

if __name__ == '__main__':
	main()
