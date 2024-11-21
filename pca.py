from PIL import Image
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
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

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
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise Exception(f"Error downloading image: {str(e)}")


def load_image(image_source):
    if isinstance(imag_source, str) and (image_source.startswith("http://") or image_source.startswith("https://")):
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

    return reconstructed, compression_ratio, explained_variance_ratio


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


def compress_with_different_components(image_source, components_list):
    """
    Compress image with different numbers of principal components and compare results.
    """
    # Load image
    original_img = load_and_prepare_image(image_source)

    results = []
    for n_components in components_list:
        start_time = time.time()

        # Compress image
        reconstructed, compression_ratio, explained_variance = compress_image(
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
            'reconstructed': reconstructed
        })

        # Plot results
        plot_results(original_img, reconstructed, n_components,
                     compression_ratio, mse, psnr, explained_variance)

    return results


def compression_results(results):
    print("\n Compression Results Summary:")
    print("-------------------------------")
    for result in results:
        print(f"\n Number of components:    {result['n_components']}")
        print(f" Compression ratio:       {result['compression_ratio']:.2f}x")
        print(f" MSE:                     {result['mse']:.2f}")
        print(f" PSNR:                    {result['psnr']:.2f} dB")
        print(f" Explained variance:      {result['explained_variance']:.2%}")
        print(f" Compression time:        {
              result['compression_time']:.2f} seconds")
        print("-------------------------------")
