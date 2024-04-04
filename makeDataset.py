import torch
import torchvision.transforms.functional as TF
import h5py
import scipy.io as io
import os
import glob
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F  # Import for GPU-compatible Gaussian filter
device = torch.device("mps")  # Use "mps" for Metal Performance Shaders on M1

# Define a GPU-compatible Gaussian filter function
def gaussian_filter_density_gpu(density):
    # Normalize density to 0-1 range for compatibility with F.conv2d
    density = density / torch.max(density)

    # Check if the density tensor has a singleton batch dimension
    if density.dim() == 4 and density.size(0) == 1:
        density = density.squeeze(0)  # Remove singleton batch dimension

    # Define the Gaussian kernel
    kernel_size = 15  # Adjust kernel size as needed
    sigma = kernel_size / 6
    gaussian_kernel = torch.exp(-(torch.arange(kernel_size) - kernel_size // 2)**2 / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Apply Gaussian filter using convolution
    density = F.conv2d(density.unsqueeze(0), gaussian_kernel.view(1, 1, -1, 1).to(density.device), padding=(kernel_size // 2, 0))
    density = F.conv2d(density, gaussian_kernel.view(1, 1, 1, -1).to(density.device), padding=(0, kernel_size // 2))

    return density.squeeze(0)  # Remove batch dimension


def main():
    # Device configuration for Mac M1 GPU
    device = torch.device("mps")  # Use "mps" for Metal Performance Shaders on M1

    root = 'ShanghaiTech'  # Corrected directory name
    part_A_train = os.path.join(root, 'part_A/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A/test_data', 'images')
    path_sets = [part_A_train, part_A_test]

    for path in path_sets:
        img_paths = glob.glob(os.path.join(path, '*.jpg'))

        for img_path in tqdm(img_paths):
            # Load ground truth using libraries optimized for potentially large MAT files
            mat_path = img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_')
            mat = io.loadmat(mat_path)
            gt = mat["image_info"][0,0][0,0][0]

            # Load image using GPU-compatible PIL transforms
            img = Image.open(img_path).convert('RGB')
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
            img_shape = img_tensor.shape[-2:]  # Get image dimensions

            # Create density map on GPU
            density_map = torch.zeros((1, img_shape[0], img_shape[1]), dtype=torch.float32, device=device)
            for i in range(len(gt)):
                x, y = int(gt[i][0]), int(gt[i][1])
                if y < img_shape[0] and x < img_shape[1]:
                    density_map[0, y, x] = 1

            # Apply Gaussian filter on GPU using GPU-compatible function
            density_map = gaussian_filter_density_gpu(density_map)

            density_map = density_map.cpu().numpy()  # Move data back to CPU for saving

            # Save density map
            save_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth')
            print("Saving density map to:", save_path)
            with h5py.File(save_path, 'w') as hf:
                hf.create_dataset('density', data=density_map)
                print("Density map saved successfully.")

    print("All densities computed.")

if __name__ == "__main__":
    main()