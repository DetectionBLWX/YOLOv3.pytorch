'''
Function:
	define the darknet
Author:
	Charles
'''
import torch
import torch.nn as nn


'''define darknet53'''
class Darknet53(nn.Module):
	def __init__(self, **kwargs):
		super(Darknet53, self).__init__()
	'''forward'''
	def forward(self, x):
		pass