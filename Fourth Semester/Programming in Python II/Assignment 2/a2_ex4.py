import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

def plot_evaluation_metrics(data: dict, figsize: tuple, save_path: str = None):
    # Extract data for Accuracy
    acc_values = data["Accuracy"]["values"]
    acc_labels = data["Accuracy"]["labels"]
    
    # Extract data for F1
    f1_values = data["F1"]["values"]
    f1_labels = data["F1"]["labels"]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)

    bp_acc = axes[0].boxplot(
        acc_values,
        patch_artist=True,  
        vert=True            
    )
    colors_acc = plt.cm.tab10(np.arange(len(acc_values)))
    for patch, c in zip(bp_acc['boxes'], colors_acc):
        patch.set_facecolor(c)
    for median in bp_acc['medians']:
        median.set_color('black')
    
    axes[0].set_xticks(np.arange(1, len(acc_values) + 1))
    axes[0].set_xticklabels(acc_labels, rotation=0)
    # Title
    axes[0].set_title("Accuracy")
    # Dashed y-axis grid
    axes[0].grid(axis='y', ls='--', alpha=0.6)
    
    bp_f1 = axes[1].boxplot(
        f1_values,
        patch_artist=True,
        vert=True
    )
    colors_f1 = plt.cm.tab10(np.arange(len(f1_values)))
    for patch, c in zip(bp_f1['boxes'], colors_f1):
        patch.set_facecolor(c)
    for median in bp_f1['medians']:
        median.set_color('black')
    
    axes[1].set_xticks(np.arange(1, len(f1_values) + 1))
    axes[1].set_xticklabels(f1_labels, rotation=0)
    axes[1].set_title("F1")
    axes[1].grid(axis='y', ls='--', alpha=0.6)
    
    for ax in axes:
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))  
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()

if __name__ == "__main__":
    with open("a2_ex4_data.pkl", "rb") as f:
        data = pkl.load(f)
    plot_evaluation_metrics(data, (3 * len(data), 4), "ex4.png")