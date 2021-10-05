import os
import sys
import concurrent.futures
import threading
from queue import Queue
from threading import Thread

K = [1, 1, 1, 5]
N = [2, 3, 4, 4]


#for i in range(len(K)):
#	os.system('python3 hw1.py --num_classes={} --num_samples={}'.format(N[i], K[i]))

# Problem 4
#a K=1, N=3, diff optimizer, learning rate
opt = ['SGD', 'RMSPROP']
lr = [0.001, 0.0001]

#b LSTM size
lstm = [128, 256, 512]

#for i in range(len(lstm)):
#	os.system('python3 hw1.py --num_classes=5 --num_samples=1 --model_size={}'.format())

class MetaLearner(Thread):
	def __init__(self, queue):
		Thread.__init__(self)
		self.queue = queue
	def run(self):
		while True:
			settings = self.queue.get()
			try:
				os.system('')
			finally:
				self.queue.task_done()
def main():
	K = [1, 1, 1, 5]
	N = [2, 3, 4, 4]
	# K = 1, N = 5
	opt = ['SGD', 'RMSPROP']
	lr = [0.001, 0.0001]
	lstm = [128, 256, 512]
	settings = 
	queue = Queue()
	for x in range(8): 
		learner = MetaLearner(queue)
		learner.daemon = True
		worker.start()
	for setting in settings:
		queue.put()
	queue.join()

if __name__ == '__main__':
	main()
