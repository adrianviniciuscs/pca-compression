from PIL import Image
import re
import requests
from io import BytesIO
import numpy as np
import time
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit_transform(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort the eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio = self.explained_variance / total_var

        # Transform the data
        X_transformed = np.dot(X_centered, self.components_)

        return X_transformed

    def inverse_transform(self, X):
        return np.dot(X, self.components_.T) + self.mean_


def download_image(url):
    try:
        headers = {
            'User-Agent': 'PCACompression/1.0 )'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Verificar o tipo de conteúdo retornado
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            raise Exception(f"Unexpected content type: {content_type}")

        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        raise Exception(f"Error downloading image: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def load_image(image_source):
    if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
        img = download_image(image_source)
    else:
        img = Image.open(image_source)

    img_gray = img.convert("L")
    img_array = np.array(img_gray)

    return img_array


def compress_image(img_array, n_components):
    height, width = img_array.shape

    X = img_array.reshape(height, width)

    pca = PCA(n_components=n_components)

    compressed_img = pca.fit_transform(X)

    reconstructed = pca.inverse_transform(compressed_img)

    original_size = X.nbytes
    compressed_size = compressed_img.nbytes + pca.components_.nbytes
    compression_ratio = compressed_size / original_size

    explained_variance_ratio = pca.explained_variance_ratio.sum()

    return reconstructed, compression_ratio, explained_variance_ratio, compressed_size


def evaluate_compression(original, compressed):
    mse = np.mean((original - compressed) ** 2)

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr


def plot_results(original, reconstructed, n_components, compression_ratio, mse, psnr, explained_variance):
    """
    Plot original and reconstructed images with compression metrics.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f'Reconstructed Image\n(n_components={n_components})')
    plt.axis('off')

    plt.suptitle(f'Image Compression Metrics:\n'
                 f'Compression Ratio: {compression_ratio:.2f}x\n'
                 f'MSE: {mse:.2f}\n'
                 f'PSNR: {psnr:.2f} dB\n'
                 f'Explained Variance: {explained_variance:.2%}')

    plt.tight_layout()
    plt.show()


def plot_all_results(original, results, original_size, cols=3):
    """
    Plot the original image and all reconstructed images in a grid, with metrics displayed.

    Parameters:
        original (np.array): The original grayscale image.
        results (list): List of dictionaries containing reconstruction results and metrics.
        original_size (int): The size of the original image in bytes.
        cols (int): Number of columns for the grid.
    """
    num_results = len(results)
    # Include the original image, calculate rows
    rows = (num_results + 1 + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))  # Adjust figure size based on rows
    # Plot original image
    plt.subplot(rows, cols, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f"Original Image\n Size: {original_size:.2f} bytes")
    plt.axis('off')

    # Plot reconstructed images
    for i, result in enumerate(results, start=2):
        reconstructed = result['reconstructed']
        n_components = result['n_components']
        compression_ratio = result['compression_ratio']
        mse = result['mse']
        psnr = result['psnr']
        explained_variance = result['explained_variance']
        compressed_size = result['compressed_size']/1024 * 1024

        plt.subplot(rows, cols, i)
        plt.imshow(reconstructed, cmap='gray')
        plt.title(
            f"{n_components} Components\n"
            f"CR: {compression_ratio:.2f}x\n"
            f"MSE: {mse:.2f}\n"
            f"PSNR: {psnr:.2f} dB\n"
            f"EV: {explained_variance:.2%}\n"
            f"Size: {compressed_size} bytes"
        )
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def compress_with_different_components(image_source, components_list):
    """
    Compress image with different numbers of principal components and compare results.
    """
    # Load image
    original_img = load_image(image_source)
    original_size = original_img.nbytes

    results = []
    for n_components in components_list:
        start_time = time.time()

        # Compress image
        reconstructed, compression_ratio, explained_variance, compressed_size = compress_image(
            original_img, n_components)

        # Evaluate compression
        mse, psnr = evaluate_compression(original_img, reconstructed)

        compression_time = time.time() - start_time

        results.append({
            'n_components': n_components,
            'compression_ratio': compression_ratio,
            'mse': mse,
            'psnr': psnr,
            'explained_variance': explained_variance,
            'compression_time': compression_time,
            'reconstructed': reconstructed,
            'compressed_size': compressed_size
        })

    # Plot results
    plot_all_results(original_img, results, original_size)

    return results


def compression_results(results):
    print("\n Compression Results Summary:")
    print("-------------------------------")
    print(f"{'Components':<20}{'Compression Ratio (%)':<25}{'MSE':<10}{
          'PSNR (dB)':<10}{'Explained Variance (%)':<25}{'Time (s)':<10}")
    print("-" * 100)
    for result in results:
        compression_ratio_percentage = result['compression_ratio'] * 100
        print(f"{result['n_components']:<20}{compression_ratio_percentage:<25.2f}{result['mse']:<10.2f}{
              result['psnr']:<10.2f}{result['explained_variance'] * 100:<25.2f}{result['compression_time']:<10.2f}")
    print("-" * 100)


if __name__ == "__main__":

    image_url = input("Please enter the URL to the image: ")

    components = [5, 10, 20, 50, 100, 200]
    results = compress_with_different_components(image_url, components)

    compression_results(results)
