import matplotlib.pyplot as plt
import pickle as pkl

def plot_runtime(data: dict, figsize: tuple, save_path: str = None):
    # Set the figure size
    plt.rcParams["figure.figsize"] = figsize
    
    # Plot lines for both algorithms
    plt.plot(data['n_instances'], data['Algorithm 1'],
             label='Algorithm 1', ls='-', color='r', marker='o')
    plt.plot(data['n_instances'], data['Algorithm 2'],
             label='Algorithm 2', ls='-', color='b', marker='s')
    
    # Title and axis labels
    plt.title("Runtime comparison between Algorithms 1 & 2")
    plt.xlabel("Number of instances")
    plt.ylabel("Runtime (seconds)")
    
    # Convert the x-ticks into “xM” format
    x_values = data['n_instances']
    x_labels = [f"{x // 1_000_000}M" for x in x_values]
    plt.xticks(x_values, x_labels)
    
    # Add a grid with dashed lines and alpha=0.6
    plt.grid(ls='-', alpha=0.6)
    
    # Add legend and adjust layout
    plt.legend()
    plt.tight_layout()
    
    # Save the figure if `save_path` is specified
    if save_path is not None:
        plt.savefig(save_path)
    
    # Show the figure
    plt.show()

if __name__ == "__main__":
    with open("a2_ex1_data.pkl", "rb") as f:
        data = pkl.load(f)
    plot_runtime(data, (7, 4), "ex1.png")
