import nibabel as nib
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from torch_radon import Radon, RadonFanbeam, ParallelBeam
import numpy as np


def plot(x):
    x = x.detach().cpu().numpy()
    plt.imshow(x)
    plt.show()

theta = np.linspace(0, np.pi, 256, endpoint=False)
theta = Tensor(theta).requires_grad_()
# theta = np.arange(360)
det_count = int(np.ceil(np.sqrt(2)*512))
sparse_radon = Radon(512, theta, det_count=det_count)

file_name = '/home/tomerweiss/ct/pd-unet-ct/big_data//train/ABD_LYMPH_042.nii.gz'
# x = Tensor(nib.load(file_name).get_fdata())[:, :, 100][None, None, :, :]
# torch.save(x, 'test_sample.pt')
x = torch.load('test_sample.pt').cuda()
x -= x.min()
x /= x.max()

plot(x[0, 0])

# forward and inverse
sino = sparse_radon.forward(x)
corrupted = sparse_radon.backprojection(sino)
plot(corrupted[0, 0])

# forward, filter in the sinogram domain and inverse
sino = sparse_radon.forward(x)
filtered_sinogram = sparse_radon.filter_sinogram(sino, "ram-lak")
corrupted = sparse_radon.backprojection(filtered_sinogram)
plot(corrupted[0, 0])

# check gradients to theta
print(f"Corrupted requires_grad: {corrupted.requires_grad}")
print(f"Theta grad before: {theta.grad}")
loss = (corrupted-x).abs().mean()
loss.backward()
print(f"Theta grad after: {theta.grad}")

