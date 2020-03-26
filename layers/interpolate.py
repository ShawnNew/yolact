import torch.nn as nn
import torch.nn.functional as F

class InterpolateModule(nn.Module):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def forward(self, x):
		import torch
		sh = torch.tensor(x.shape)
		return F.interpolate(x, size=(sh[2] * 2, sh[3] * 2), mode='nearest')
