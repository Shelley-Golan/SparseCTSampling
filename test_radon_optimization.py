import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch.optim import Adam

from test_radon_scikit import mask_circle, custom_radon, custom_iradon, to_np

plot = True
filter = 'ramp'
image = torch.load('test_sample.pt').squeeze()
image -= image.min()
image /= image.max()
image = mask_circle(image)
image = torch.tensor(image, requires_grad=False).cuda()

# theta = np.linspace(0., 180., max(image.shape) // 40, endpoint=False)   # uniform init
theta = np.random.rand(20)*180                                          # random init
torch_theta = torch.tensor(theta).cuda().requires_grad_(True)

optimizer = Adam([torch_theta], lr=1e-1)

for i in range(301):
    custom_sinogram = custom_radon(image.unsqueeze(0).unsqueeze(0), theta=torch_theta)
    reconstruction_torch = custom_iradon(custom_sinogram,
                                         theta=torch_theta,
                                         filter_name=filter).squeeze()
    loss = ((reconstruction_torch - image)**2).mean()

    if i % 50 == 0:
        title = f"{i} - Loss: {loss:.4f}, MAE: {(reconstruction_torch - image).abs().mean():.4f}"
        print(title)
        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(9, 2.7))
            fig.suptitle(title)
            axes[0].set_title('Ground truth')
            axes[0].imshow(to_np(image), cmap='gray')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            # axes[1].set_title('Reconstruction')
            axes[1].imshow(to_np(reconstruction_torch), cmap='gray')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[2].set_title('Thetas')
            t = to_np(torch.deg2rad(torch_theta))
            axes[2].scatter(np.sin(t), np.cos(t), cmap='gray')
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            axes[2].set_aspect('equal')
            fig.show()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
