import os
import sys

K = [1, 1, 1] #5]
N = [2, 3, 4] #4]

for i in range(len(K)):
	os.system('python3 hw1.py --num_classes={} --num_samples={} --logdir=experiment'.format(N[i], K[i]))

