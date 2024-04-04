from dataset import ShanghaiTechDataset
import matplotlib.pyplot as plt

def main():
    # Instantiate the dataset
    dataset = ShanghaiTechDataset(root='ShanghaiTech', part='part_B', phase='train')
    
    # Load an example
    image, density = dataset[199]  # Just as an example, loading the first item
    
    image = image.numpy().transpose(1, 2, 0)
    density = density.numpy().squeeze()

    # Displaying the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    # Displaying the corresponding density map
    plt.subplot(1, 2, 2)
    plt.imshow(density, cmap='jet')
    plt.title('Density Map')
    plt.show()

if __name__ == '__main__':
    main()
