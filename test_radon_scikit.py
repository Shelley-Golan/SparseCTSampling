import numpy as np
from time import time
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from kornia.geometry.transform import warp_affine, get_rotation_matrix2d
import pydicom as dicom
import matplotlib.image as mpimg
from skimage import color

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon



def get_rot_mat(theta, center):
    cos_a, sin_a = torch.cos(theta), -torch.sin(theta)
    return torch.stack([
        torch.stack([cos_a, sin_a, -center * (cos_a + sin_a - 1)]),
        torch.stack([-sin_a, cos_a, -center * (cos_a - sin_a - 1)])
    ])

def rot_img(x, theta, dtype, center):
    rot_mat = get_rot_mat(theta, center)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    # grid = F.affine_grid(rot_mat, x.size(), align_corners=False).type(dtype)
    # return F.grid_sample(x, grid, align_corners=False)
    return warp_affine(x, rot_mat, x.shape[-2:])


def custom_radon(image, theta=None):
    if theta is None:
        theta = torch.arange(180, requires_grad=True)
    # padded_image = convert_to_float(image, preserve_range)
    center = image.shape[-1]//2
    sums = []
    for i, angle in enumerate(torch.deg2rad(theta)):
        rotated = rot_img(image, angle, image.dtype, center)
        sums += [rotated.sum(2).unsqueeze(-1)]
    radon_image = torch.cat(sums, dim=-1)
    return radon_image

def custom_get_fourier_filter(size, filter_name):
    n = torch.from_numpy(np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int))))
    f = torch.zeros((size))
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n) ** 2

    fourier_filter = 2 * torch.real(torch.fft.fft(f))
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = torch.pi * torch.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega
    elif filter_name == "cosine":
        freq = torch.tensor(np.linspace(0, np.pi, size, endpoint=False))
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= torch.fft.fftshift(torch.tensor(np.hamming(size)))
    elif filter_name == "hann":
        fourier_filter *= torch.fft.fftshift(torch.tensor(np.hanning(size)))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter

def custom_interp_batch(x, xp, fp):
    # based on implementation taken from: https://github.com/pytorch/pytorch/issues/50334
    m = ((fp[:, :, 1:] - fp[:, :, :-1]) / (xp[:, :, 1:] - xp[:, :, :-1]))
    b = (fp[:, :, :-1] - (m * xp[:, :, :-1]))
    b = b.reshape(x.shape[0], -1)
    step_size = m.shape[-1]
    m = m.reshape(x.shape[0], -1)

    indices = torch.sum(torch.ge(x.reshape(x.shape[0], -1)[:, :, None], xp.reshape(x.shape[0], -1)[:, None, :]), 2) - 1
    indices = torch.clamp(indices, 0, m.numel() / x.shape[0] - 1)
    delta = torch.zeros((1, x.shape[-1] ** 2), dtype=indices.dtype)
    to_add = []
    for i in range(x.shape[0]):
        to_add += [delta.clone()]
        delta += step_size
    to_add = torch.cat(to_add, dim=0)
    indices += to_add
    m = torch.take(m, indices).reshape(x.shape)
    b = torch.take(b, indices).reshape(x.shape)

    return m * x + b

def lin_interpolate(data, x):
    x_shape = x.shape
    x = x.view(-1)

    n = data.shape[-1]
    mask=torch.lt(x,n).float()
    x = x.clone()*mask
    idx = torch.floor(x)
    frac = x - idx

    left = data[..., idx.long()]
    mask2=torch.ne(idx,n-1).float()
    idx=idx.clone() * mask2
    right = data[..., idx.long() + 1]
    output=(1.0 - frac) * left + frac * right
    return (output*mask*mask2).view(*(data.shape[:-1]+x_shape))

def custom_sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[2]))
    pad = diagonal - sinogram.shape[2]
    old_center = sinogram.shape[2] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = (0, 0, pad_before, pad - pad_before)
    return F.pad(sinogram, pad_width, mode='constant', value=0)

def custom_iradon(radon_image, theta=None, filter_name="ramp",
                  circle = False, interpolation="linear"):
    if theta is None:
        theta = torch.arange(180, requires_grad=True)
    angles_count = theta.shape[0]
    if angles_count != radon_image.shape[3]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter_name not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    # radon_image = convert_to_float(radon_image, preserve_range)
    dtype = radon_image.dtype

    output_size = radon_image.shape[2]
    if circle:
        radon_image = custom_sinogram_circle_to_square(radon_image)
    img_shape = radon_image.shape[2]
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = (0, 0, 0, projection_size_padded - img_shape)
    img = F.pad(radon_image, pad_width, mode='constant', value=0)

    # Apply filter in Fourier domain
    filter = custom_get_fourier_filter(projection_size_padded, filter_name).to(img.device)
    projection = torch.fft.fft(img, dim=2) * filter[None, None, :, None]
    radon_filtered = torch.real(torch.fft.ifft(projection, dim=2)[:, :, :img_shape, :])

    # Reconstruct image by interpolation
    reconstructed = torch.zeros((radon_image.shape[0], radon_image.shape[1], output_size, output_size),
                             dtype=dtype, device=img.device)
    radius = output_size // 2
    xpr, ypr = torch.meshgrid(torch.arange(output_size), torch.arange(output_size), indexing='ij')
    xpr = xpr.to(img.device) - radius
    ypr = ypr.to(img.device) - radius
    x = torch.arange(img_shape)
    x = x[None, None, :].repeat(reconstructed.shape[0], 1, 1)
    for i, angle in zip(range(radon_filtered.shape[3]), torch.deg2rad(theta)):
        col = radon_filtered.transpose(2, 3)[:, :, i, :]
        t = (ypr * torch.cos(angle) - xpr * torch.sin(angle)) + img_shape // 2
        interp_res = lin_interpolate(col, t)
        reconstructed += interp_res

    if circle:
        out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
        reconstructed[:,:,out_reconstruction_circle] = 0.

    return (reconstructed * torch.pi / (2 * angles_count))


def mask_circle(image):
    shape_min = min(image.shape)
    radius = shape_min // 2
    img_shape = np.array(image.shape)
    coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]],
                      dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    image[outside_reconstruction_circle] = 0
    return image

def to_np(x):
    return x.detach().cpu().numpy()

if __name__ == '__main__':
    # image = shepp_logan_phantom()
    # image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)
    #ds = dicom.dcmread('pd-unet-ct/big_data/not_circle/test/10072.dcm')
    image = mpimg.imread('test_sample.jpg')
    image = color.rgb2gray(image)
    #ds = dicom.dcmread('0.dcm')
    #image = torch.from_numpy(ds.pixel_array).float()
    image -= image.min()
    image /= image.max()
    #image = mask_circle(image)
    image = torch.tensor(image, requires_grad=False).cuda()
    x=image.shape
    circle = False
    sparsity = 16
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5))

    ax1.set_title("Original")
    ax1.imshow(to_np(image), cmap=plt.cm.Greys_r)
    test_batch = False
    theta = np.arange(180)[::sparsity].astype(float)
    #theta = np.linspace(0,180,180//4, endpoint=False)
    #theta = np.linspace(0., 180., 256, endpoint=False)
    torch_theta = torch.tensor(theta, requires_grad=True).cuda()

    s = time()
    #sinogram = radon(to_np(image), theta=theta)
    print(f'time randon original:{time()-s:.3f}s')

    s = time()
    if test_batch:
        custom_sinogram = custom_radon(image.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1), theta=torch_theta)
    else:
        custom_sinogram = custom_radon(image.unsqueeze(0).unsqueeze(0), theta=torch_theta)
    print(f'time randon custom:{time()-s:.3f}s')

    #custom_sinogram.sum().backward()

    filter = 'ramp'
    s = time()
    #reconstruction_fbp = iradon(sinogram, theta=theta, filter_name=filter, circle=circle)
    print(f'time irandon original:{time()-s:.3f}s')
    s = time()
    reconstruction_torch = custom_iradon(custom_sinogram, theta=torch_theta, filter_name=filter, circle=circle)
    print(f'time irandon custom:{time()-s:.3f}s')
    #reconstruction_torch = custom_iradon(torch.from_numpy(sinogram).unsqueeze(0).unsqueeze(0), theta=torch_theta, filter_name='ramp')
    if test_batch:
        reco_error = reconstruction_fbp - to_np(reconstruction_torch[1].squeeze(0))
        sino_error = sinogram - to_np(custom_sinogram[1, :, :, :].squeeze(0))
        #reconstruction_torch.sum().backward()
        #print("theta grad:")
        #print(torch_theta.grad)
        reconstruction_torch = reconstruction_torch[0].unsqueeze(0)
        custom_sinogram = custom_sinogram[0].unsqueeze(0)
        a = custom_sinogram.shape
        print(f'custom sino error: {np.sqrt(np.mean(sino_error ** 2)):.3g}')
        print(f'custom reco error: {np.sqrt(np.mean(reco_error ** 2)):.3g}')

    #reco_error = reconstruction_fbp - to_np(reconstruction_torch.squeeze(0).squeeze(0))
    #sino_error = sinogram - to_np(custom_sinogram.squeeze(0).squeeze(0))
    #error = reconstruction_fbp - to_np(image)
    #error_custom = to_np(reconstruction_torch.squeeze(0).squeeze(0)) - to_np(image)

    #print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error ** 2)):.3g}')
    #print(f'custom FBP rms reconstruction error: {np.sqrt(np.mean(error_custom ** 2)):.3g}')
    #print(f'custom sino error: {np.sqrt(np.mean(sino_error ** 2)):.3g}')
    #print(f'custom reco error: {np.sqrt(np.mean(reco_error ** 2)):.3g}')

    #dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    #ax2.set_title("Radon transform\n(Sinogram)")
    #ax2.set_xlabel("Projection angle (deg)")
    #ax2.set_ylabel("Projection position (pixels)")
    #ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
    #           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    #           aspect='auto')
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / custom_sinogram.shape[0]
    ax3.set_title("custom Radon transform\n(Sinogram)")
    ax3.set_xlabel("Projection angle (deg)")
    ax3.set_ylabel("Projection position (pixels)")
    ax3.imshow(to_np(custom_sinogram.squeeze(0).squeeze(0)), cmap=plt.cm.Greys_r,
               extent=(-dx, 180.0 + dx, -dy, custom_sinogram.shape[0] + dy),
               aspect='auto')
    #reconstruction_torch.sum().backward()
    #print("theta grad:")
    #print(torch_theta.grad)
    #print("image grad:")
    #print(image.grad)

    fig.tight_layout()
    plt.show()

    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                   sharex=True, sharey=True)
    #ax1.set_title("Reconstruction\nFiltered back projection")
    #ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    #ax2.set_title("Reconstruction error\nFiltered back projection")
    #ax2.imshow(reconstruction_fbp - to_np(image), cmap=plt.cm.Greys_r, **imkwargs)
    #plt.show()
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
    #                               sharex=True, sharey=True)
    ax1.set_title("torch Reconstruction\nFiltered back projection")
    ax1.imshow(to_np(reconstruction_torch.squeeze(0).squeeze(0)), cmap=plt.cm.Greys_r)
    ax2.set_title("torch Reconstruction error\nFiltered back projection")
    ax2.imshow(to_np(reconstruction_torch.squeeze(0).squeeze(0)) - to_np(image), cmap=plt.cm.Greys_r, **imkwargs)
    plt.show()