'''
Function:
	generate anchors by using k-means clustering
Author:
	Charles
'''
import os
import random
import numpy as np


'''anchor generator by using k-means clustering'''
class AnchorGenerator(object):
	def __init__(self, cfg, **kwargs):
		self.cfg = cfg
	'''generate'''
	def generate(self):
		pass
	'''calc ious between gt and anchors'''
	def calcIoU(self):
		pass
	'''save results'''
	def save(self, anchors, savepath='anchors.txt'):
		f = open(savepath)
		for anchor in anchors: f.write(str(anchor[0])+str(anchor[1])+'\n')
		f.close()


'''run'''
if __name__ == '__main__':
	cfg = {
			'image_size': {'long_side': 1333, 'short_side': 800},
			'annfilepath': 'coco2017/annotations/instances_train2017.json',
			'num_anchors': 9
		}
	generator = AnchorGenerator(cfg)
	generator.generate()