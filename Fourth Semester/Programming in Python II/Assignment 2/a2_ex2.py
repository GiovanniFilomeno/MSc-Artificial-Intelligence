import matplotlib.pyplot as plt
import numpy as np

def plot_scores(data: np.ndarray, figsize: tuple, save_path: str = None):
    n_scores = data.size

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)

    bins_left = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    axes[0].hist(data,
                 bins=bins_left,
                 color='blue',
                 edgecolor='black')
    axes[0].set_title(f"Histogram of {n_scores} exam scores in ranges of 10")
    axes[0].set_xlabel("Scores")
    axes[0].set_ylabel("# Students")  
    axes[0].grid(axis='y', ls='--', alpha=0.6)  

    bins_right = [0, 50, 62.5, 75, 87.5, 100]
    axes[1].hist(data,
                 bins=bins_right,
                 color='red',
                 edgecolor='black')
    axes[1].set_title(f"Histogram of {n_scores} exam scores in 5-level grading")
    axes[1].grid(axis='y', ls='--', alpha=0.6)  

    bin_centers = [(bins_right[i] + bins_right[i+1]) / 2 for i in range(len(bins_right)-1)] # center of bin
    grade_labels = ["Unsatisfactory", "Adequate", "Satisfactory", "Good", "Very Good"]
    axes[1].set_xticks(bin_centers)
    axes[1].set_xticklabels(grade_labels, rotation=45, ha='center')

    plt.tight_layout()

    # Save the figure if a path is given
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

if __name__ == "__main__":
    data = np.loadtxt("a2_ex2_data.csv", delimiter=",", skiprows=1)
    plot_scores(data, (10, 5), "ex2.png")