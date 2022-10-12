import nibabel as nib
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from torch_radon import Radon, RadonFanbeam
import numpy as np


def plot(x):
    x = x.detach().cpu().numpy()
    plt.imshow(x)
    plt.show()

theta = np.linspace(0, np.pi, 256, endpoint=False)
theta = Tensor(theta).requires_grad_()
geom = {'source_distance':400., 'det_distance':150., 'det_count':511, 'det_spacing':1.0}
sparse_radon = RadonFanbeam(
    resolution=512,
    angles=theta,
    source_distance=geom['source_distance'],
    det_distance=geom['det_distance'],
    det_count=geom['det_count'],
    det_spacing=geom['det_spacing'],
)
file_name = '/home/tomerweiss/ct/pd-unet-ct/big_data//train/ABD_LYMPH_042.nii.gz'
x = Tensor(nib.load(file_name).get_fdata())[:, :, 100][None, None, :, :]
torch.save(x, 'test_sample.pt')
# x = torch.load('test_sample.pt').cuda()
# x = x.cuda()
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
