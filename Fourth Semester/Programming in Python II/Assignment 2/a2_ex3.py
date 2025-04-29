import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

def plot_distribution(data: dict, figsize: tuple, save_path: str = None):
    # Extract plant coordinate arrays
    plant_a = data['Plant A']
    plant_b = data['Plant B']
    plant_c = data['Plant C']
    
    plt.figure(figsize=figsize)
    
    # Scatter for Plant A 
    plt.scatter(
        plant_a[:, 0],
        plant_a[:, 1],
        marker='o',
        color='red',
        alpha=0.4,
        label=f"Plant A: {plant_a.shape[0]} istances"
    )
    
    # Scatter for Plant B 
    plt.scatter(
        plant_b[:, 0],
        plant_b[:, 1],
        marker='x',
        color='green',
        alpha=0.4,
        label=f"Plant B: {plant_b.shape[0]} istances"
    )
    
    # Scatter for Plant C 
    plt.scatter(
        plant_c[:, 0],
        plant_c[:, 1],
        marker='s',
        color='blue',
        alpha=0.4,
        label=f"Plant C: {plant_c.shape[0]} istances"
    )
    
    # Axis labels, title, legend
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    plt.title("Distribution of 3 plants: A, B and C")
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure if `save_path` is specified
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    with open("a2_ex3_data.pkl","rb") as f:
        data = pkl.load(f)
    plot_distribution(data, (10, 6), "ex3.png")
