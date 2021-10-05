import os
import sys

K = [1, 1, 1, 5]
N = [2, 3, 4, 4]


for i in range(len(K)):
	os.system('python3 hw1.py --num_classes={} --num_samples={}'.format(N[i], K[i]))

# Problem 4
#a K=1, N=3, diff optimizer, learning rate
opt = ['SGD', 'RMSPROP']
lr = [0.001, 0.0001]

#b LSTM size
lstm = [128, 256, 512]

for i in range(len(lstm)):
	os.system('python3 hw1.py --num_classes=5 --num_samples=1 --model_size={}'.format())
