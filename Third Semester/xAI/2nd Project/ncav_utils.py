import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from ConceptExtractor import ConceptExtractor


def calc_concept_number(thresholds: list, A, init_concepts):
    concept_dict_all_classes = {  # ideal number of concepts for each threshold
        0.1: [None, None],  # [nr concepts, P]
        0.2: [None, None],
        0.3: [None, None],
        0.4: [None, None],
        0.5: [None, None],
        0.6: [None, None],
        0.7: [None, None]
    }

    for threshold in thresholds:
        print(f"threshold: {threshold}")
        ConceptExtractor_activations = ConceptExtractor(A, seed=1234)
        concepts = init_concepts
        similar_concepts = True
        while similar_concepts and concepts > 0:
            # train NMF
            ConceptExtractor_activations.apply_NMF(nr_concepts=concepts)
            # Extract the P matrix where the rows correspond to the CAVs
            P = ConceptExtractor_activations.P
            # Calculate the cosine similarity between the CAVs
            cos_sim = cosine_similarity(P)
            # Extract the upper triangle, excluding the diagonal to get the similarities beteen CAVs
            cos_sim = np.triu(cos_sim, k=1)
            # check if all the CAVs are below the threshold
            above_threshold = cos_sim > threshold
            if above_threshold.sum() == 0:
                similar_concepts = False
                concept_dict_all_classes[threshold][0] = concepts
                concept_dict_all_classes[threshold][1] = P
            else:
                concepts -= 1
    return concept_dict_all_classes


def TCAV_directional_derivative(activation_layer: np.array, layer_name: int, cav_vector: np.array, epsilon: float, model, class_k: int) -> np.array:
    """
    function calculates the directional derivatives which measure the sensitivity of predictions of the model of class k
    w.r.t. the cav_vector
    Note: we assume for function inputs that activation_layer and cav_vector are calculated for class k and not mixed and
    that all parameters that are influenced by layer_name use the same layer
    Note: this function is only suitable for one image-batch
    :param activation_layer: layer retrieved from the model, has activations for class k,
    shape 4dim: (batch_size, height, width, channels)
    :param layer_name: name of activation_layer
    :param cav_vector: holds the a concept, shape: 1dim (channels)
    :param epsilon: parameter for the directional derivative, should be small
    :param model: model used to calculate the predictions, should be an intance with pretrained weights!
    :param class_k: index of the cth class [0:1000] as model outputs one of the 1000 ImageNet classes
    :return: directional_derivatives_C_l: the directional derviative for cav C from layer l as np.array of size: batch_size
    """
    model.eval()
    with torch.no_grad():
        input1 = activation_layer + epsilon*cav_vector
        # not correct!! :batch_size, height, width, channels = activation_layer.shape
        # x = activation_layer.reshape(batch_size, channels, height, width)
        input1 = np.transpose(input1, (0, 3, 1, 2)) 
        logit1 = model.forward_from_layer(torch.Tensor(input1), layer_name).detach().numpy()[:,class_k]
        input2 = np.transpose(activation_layer, (0, 3, 1, 2)) 
        logit2 = model.forward_from_layer(torch.Tensor(input2), layer_name).detach().numpy()[:,class_k]
    return (logit1 - logit2) / epsilon

def TCAV_score(directional_derivatives_C_l: np.array) -> float:
    """
    function that calculates the fraction of images from class k where the concept C positively influenced the image
    being classified as k
    Note: we assume for all inputs that only class k was used to get actionvations, cavs,... otherwise the ouput
    is not meaningful
    Note: this function is only suitable for one image-batch
    :param directional_derivatives_C_l: I refer to information in TCAV_directional_derivative function
    :return: score, float between 0 and 1
    """
    return np.sum(directional_derivatives_C_l > 0) / directional_derivatives_C_l.shape[0]

def TCAV(activation_layer: np.array, layer_name: int, P: np.array, epsilon: float, model, class_k) -> np.array:
    """
    function that combines TCAV_directional_derivative and TCAV_score and returns TCAV scores for all cavs
    for parameter documentation, I refer to the function documentations
    Note: this function is only suitable for one image-batch
    :param P: is a matrix of shape (c', c) = (concept, channels) retrieved from NMF and holding the cav vectors
    :param class_k: index of the cth class [0:1000] as model outputs one of the 1000 ImageNet classes
    :return: TCAV scores for all conecpts Ci for images of class k from layer l
    """
    scores = np.zeros(P.shape[0])
    for concept_idx in range(P.shape[0]):
        cav_vector = P[concept_idx].reshape(-1)
        derivatives = TCAV_directional_derivative(activation_layer, layer_name, cav_vector, epsilon, model, class_k)
        #print("derivative: \n", derivatives.shape, "\n", derivatives)
        scores[concept_idx] = TCAV_score(derivatives)
        #print(f"concept: {concept_idx}: {scores[concept_idx]}")
    return scores


def calc_avg_concept_presence(top_n, S_list, dataloader, ConceptExtractor):
    # Initialize lists to accumulate data across batches
    all_avg_concept_presence = []

    # Iterate over all batches to collect data
    for batch_idx, (data, target) in enumerate(dataloader):
        S = S_list[batch_idx].copy()
        S = S.reshape((
            data.shape[0],
            ConceptExtractor.height,
            ConceptExtractor.width,
            ConceptExtractor.nr_concepts
        ))

        # Calculate the average concept presence for the current batch
        avg_concept_presence = np.mean(S, axis=(1, 2))  # Shape (batch_size x nr_concepts)

        # Accumulate filenames and corresponding avg concept presence
        all_avg_concept_presence.append(avg_concept_presence)

    # Convert the accumulated concept presence data to a single numpy array
    # print(len(all_avg_concept_presence))
    # print(all_avg_concept_presence[0].shape)
    all_avg_concept_presence = np.vstack(all_avg_concept_presence)  # Shape (total_samples x nr_concepts)
    # print("")
    # print(all_avg_concept_presence.shape)

    # Dictionary to store the top n filenames and concept presence for each concept
    top_n_imgs = {}

    # Calculate the top n indices globally for each concept
    for k in range(all_avg_concept_presence.shape[1]):  # Iterate over each concept (channel)
        sorted_indices_high = np.argsort(all_avg_concept_presence[:, k])[::-1]
        sorted_indices_low = np.argsort(all_avg_concept_presence[:, k])

        # Get the top n indices with the highest and lowest values
        top_n_indices_high = sorted_indices_high[:top_n]
        top_n_indices_low = sorted_indices_low[:top_n]

        # Store the top n filenames and their corresponding avg concept presence for both high and low
        top_n_imgs[f"concept_{k}"] = {
            'top_n_high': {
                'avg_concept_presence': [all_avg_concept_presence[idx, k] for idx in top_n_indices_high],
                'indices': top_n_indices_high
            },
            'top_n_low': {
                'avg_concept_presence': [all_avg_concept_presence[idx, k] for idx in top_n_indices_low],
                'indices': top_n_indices_low
            }
        }

    return top_n_imgs


def plot_concept_images(concept_dict, dataloader, S=None, threshold=0.5, background_alpha=0.6):
    """
    Plots images for each concept showing the top 5 highest and lowest concept presence.

    Args:
    - concept_dict: Dictionary containing concept presence and indices.
    - dataloader: PyTorch dataloader containing the images.
    - S: concept strengths for all images
    - threshold: threshold for concept strengths
    - background_alpha: alpha of image regions without concept presence
    """
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for concept_idx, (concept, data) in enumerate(concept_dict.items()):
        # Extract high and low indices
        high_indices = data['top_n_high']['indices']
        low_indices = data['top_n_low']['indices']

        # Fetch corresponding images from dataloader
        high_images = [dataloader.dataset[idx][0] for idx in high_indices]
        high_labels = [dataloader.dataset[idx][1] for idx in high_indices]
        low_images = [dataloader.dataset[idx][0] for idx in low_indices]
        low_labels = [dataloader.dataset[idx][1] for idx in low_indices]

        if S is not None:
            high_concepts = [S[idx][concept_idx] for idx in high_indices]
            low_concepts = [S[idx][concept_idx] for idx in low_indices]

        # Start plotting
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"Images for Concept {concept_idx + 1}", fontsize=16)

        # Plot top 5 high concept presence images
        for i, (img, label) in enumerate(zip(high_images, high_labels)):
            ax = axes[0, i]
            img = img / 2 + 0.5
            img = img.permute(1, 2, 0)
            #ax.imshow(img)  # Convert from CHW to HWC
            if S is not None:
                concept = high_concepts[i].copy()
                concept = np.clip(concept, 0, 1)
                mask = concept > threshold
                highlighted_regions = np.zeros_like(concept)
                highlighted_regions[mask] = concept[mask]
                cmap = plt.cm.viridis
                heatmap = cmap(highlighted_regions)
                heatmap_rgb = heatmap[..., :3]
                # Blend the heatmap with the RGB image (using only RGBA channels)
                #blended_image = (1 - background_alpha) * img + background_alpha * heatmap[mask, :3]
                blended_image = img.numpy().copy()
                blended_image[mask] = heatmap_rgb[mask]
                ax.imshow(blended_image)
            else:
                ax.imshow(img)  # Convert from CHW to HWC
            ax.axis('off')
            ax.set_title(f"High {i + 1} - {classes[label]}")

        # Plot top 5 low concept presence images
        for i, (img, label) in enumerate(zip(low_images, low_labels)):
            ax = axes[1, i]
            img = img / 2 + 0.5
            img = img.permute(1, 2, 0)
            ax.imshow(img)  # Convert from CHW to HWC
            if S is not None:
                concept = low_concepts[i].copy()
                concept = np.clip(concept, 0, 1)
                mask = concept > threshold
                highlighted_regions = np.zeros_like(concept)
                highlighted_regions[mask] = concept[mask]
                cmap = plt.cm.viridis
                heatmap = cmap(highlighted_regions)
                # Blend the heatmap with the RGB image (using only RGBA channels)
                blended_image = img.numpy().copy()
                blended_image[mask] = heatmap_rgb[mask]
                ax.imshow(blended_image)
                # ax.imshow(np.ma.masked_where(~mask, img), origin='lower', aspect='auto', interpolation='nearest', cmap='viridis', alpha=1)
                # ax.imshow(highlighted_regions, origin='lower', aspect='auto', interpolation='nearest', cmap='viridis', alpha=0.5)
            else:
                ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Low {i + 1} - {classes[label]}")

        # Show plot for the current concept
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
