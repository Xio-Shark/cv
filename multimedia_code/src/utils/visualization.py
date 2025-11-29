import matplotlib.pyplot as plt
import numpy as np

def plot_results(images, titles, output_path, figsize=(15, 10)):
    """
    Plot multiple images in a grid.
    
    Args:
        images: List of numpy arrays (images).
        titles: List of title strings.
        output_path: Path to save the plot.
        figsize: Figure size.
    """
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_histogram(hist, title, output_path):
    """
    Plot a histogram.
    """
    plt.figure()
    plt.bar(range(256), hist, width=1)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()

def plot_line_chart(x, y, title, xlabel, ylabel, output_path, log_x=False):
    """
    Plot a line chart.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if log_x:
        plt.xscale('log')
    
    # Add value labels
    for i, txt in enumerate(y):
        plt.annotate(f"{txt:.2f}", (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.savefig(output_path)
    plt.close()
