###############################################################################
# Feed Forward Network 			   											  #
# Two layer network with ReLU activation at first and softmax at second layer #
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
	def __init__(self,in_dim,out_dim,labels):
		""" Constructor
		Input: in_dim	- Dimension of input vector
			   out_dim	- Dimension of output vector
			   vocab	- Vocabulary of the embedding
		"""
		super(FeedForward, self).__init__()
		self.fc1 = nn.Linear(in_dim,out_dim)
		self.drop = torch.nn.Dropout(0.2)
		self.fc2 = nn.Linear(out_dim,labels)
		#self.soft_max = torch.nn.Softmax(dim=1)


	def forward(self, inp):
		""" Function for forward pass
		Input:	inp 	- Input to the network of dimension in_dim
		Output: output 	- Output of the network with dimension vocab
		"""
		out_intermediate = F.relu(self.fc1(inp))
		output =self.fc2(out_intermediate)
		return output
