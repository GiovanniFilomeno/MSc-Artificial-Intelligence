[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/osI6zUIF)

# Explainable AI Assignment 2 - Model Explanations
In this assignment, you are challenged to explain a model. For this, you will research exisiting approaches and apply them to your model and interpret the results.

[The video to the presentation can be found under this link.](https://drive.google.com/file/d/1pcn1lTkddtSEthg1P_OqJfsPoBRdoxis/view?usp=sharing)


## General Information Submission

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Team Name:** SIGMA X

**Group Members**

| Student ID    | First Name  | Last Name      | E-Mail | Workload [%]  |
| --------------|-------------|----------------|--------|---------------|
| K12119148        | Moritz      | Riedl         |k12119148@students.jku.at  |25%         |
| K12105068        | Verena      | Szojak         |verena@mail.at  |25%         |
| K12315325        | Giovanni      | Filomeno         |giovanni.filomeno.30@gmail.com  |25%         |
| K12105021        | Aaron      | Zettler         |zettler.aaron@gmail.com  |25%         |

**Dataset**

The dataset used for this assignment is the **CIFAR-10**, a collection of 60,000 32x32 color images in 10 classes including: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is balanced, with an equal number of images per class, ensuring no bias towards any specific category. It has 3 channels (RGB), ensuring color information is retained.

#### **Preprocessing Steps:**
1. **Normalizazion**: Standardizing pixel values.
2. **Splitting**: Dataset divided into training, validation, and test sets. The split is 50,000 images for the training and 10,000 for the test set. The validation is derived from the training set.  
3. **Batch Size**: A batch size of 1 is used for interpretability techniques.

**Model**

The model used for this project is a custom-built **Convolutional Neural Network (CNN)**, trained on the CIFAR-10 dataset. The CNN architecture consists of the following components:

1. **Convolutional Layers**:
   - Extract spatial features using multiple convolution operations.
   - Utilize **Batch Normalization** for stable and efficient training.
   - Activation functions: **ReLU** (Rectified Linear Unit) for non-linearity.
   - **Pooling layers** (MaxPooling) to reduce spatial dimensions while preserving important features.
   - **Dropout layers** for regularization to prevent overfitting.

2. **Fully Connected Layers**:
   - Flatten the high-level feature maps from convolutional layers into a one-dimensional vector.
   - Dense layers with **ReLU activation** to model complex relationships between features.
   - Final output layer with 10 neurons (one per CIFAR-10 class), using a **softmax activation function** to produce class probabilities.

3. **Hyperparameters**:
   - **Learning rate** and **batch size** were optimized to ensure convergence.
   - The **Adam optimizer** was used for efficient gradient updates.

4. **Performance**:
    - Train accuracy: 
    - Test accuracy: 82.59%

#### Research Questions
- What parts of an image are important to classify an image? 
  - intuitively one might assume that a CNN (like humans) looks directly at the object 
  for identification but we will verify this ourselves. 
- What patterns/structures has our model learned? 
- The test accuracy of our model is at 82.59% which is not bad/good either. What 
difficulties does the CNN have when correctly predicting specific images/classes?


## Report

### Model Explanation Technique 1: Saliency Maps

Saliency maps are a gradient-based explainability method used to identify which parts of an input image contribute the most to the model's decision. By computing the gradient of the output class score with respect to the input pixels, these maps highlight regions with high importance for the classification. However, their granularity and effectiveness can be influenced by factors like image resolution and model architecture.

#### **Interpretation of Results**

1. **Key Observations**:
   - The saliency maps successfully highlight regions corresponding to the object in the image (e.g., car, bird, or plane).
   - Due to the low resolution of the CIFAR-10 images (32x32 pixels), fine details are hard to distinguish, leading to blurred importance regions.
   - The method is effective for identifying general regions of importance but lacks detailed insights into specific learned features.

2. **Limitations**:
   - The low resolution of the CIFAR-10 dataset restricts the clarity and interpretability of saliency maps.
   - As a gradient-based method, saliency maps can sometimes be noisy or fail to capture complex patterns.
   - They serve as a good starting point but should ideally be complemented with other techniques like Grad-CAM or integrated gradients for richer explanations.

3. **Utility**:
   - Provides a quick visual check of model behavior.
   - Helps debug whether the model is focusing on meaningful parts of the input or irrelevant noise.

#### **WHY**
Saliency maps are used for interpretability in deep learning models. They help understand where the model "looks" when making decisions, ensuring that it learns meaningful patterns. This aids debugging, improves trust in predictions, and provides insights into model behavior.

#### **WHO**
- **Model Developers**: To validate and debug models.
- **Model Users**: To understand predictions and ensure reliability.
- **Non-experts**: To gain intuitive insights into machine learning models via visual explanations.

#### **WHAT**
- **Visualized Data**: Gradients of class scores with respect to input pixels.
- **Focus**: Highlights regions important for predictions, providing a glimpse of the model's attention mechanism.

#### **HOW**
- **Process**:
  - Compute gradients for the output score of the predicted or true class.
  - Visualize absolute gradients as heatmaps or overlays.
- **Tools Used**: PyTorch, CIFAR-10 dataset, and visualization libraries like Matplotlib.

#### **WHEN**
- **During Training**: To validate whether the model is focusing on relevant parts of the input.
- **After Training**: To interpret predictions, debug issues, and communicate results effectively.

#### **WHERE**
- **Applications**:
  - Research settings to analyze and debug neural networks.
  - Practical domains like medical imaging or autonomous vehicles, where understanding model behavior is critical.
  - For CIFAR-10: Educational or experimental use cases to explain simple image classifiers.

#### **Conclusion**
While saliency maps provide a foundational insight into model behavior, their explanation power is limited, especially with low-resolution datasets like CIFAR-10. They serve as a valuable first step in understanding model decisions and can be augmented with other interpretability techniques for deeper insights.

### Model Explanation Technique 2: SHAP (SHapley Additive exPlanations)
SHAP values are a unified framework for interpretability, providing pixel-level contributions to model predictions. By applying a game-theoretic approach, SHAP identifies how each pixel (or feature) influences the model's decision, offering both local and global explanations. This is particularly useful for understanding neural networks trained on image data like CIFAR-10.

#### **Interpretation of Results**
1. **Key Observations**:
   - Highlights individual pixel contributions to the prediction.
   - Visualizes the importance of image regions as heatmaps.

2. **Limitations**:
   - Computationally expensive, especially for large datasets or complex models.
   - Explanations can sometimes focus on noise or irrelevant regions if the model itself is poorly trained.
   - The resolution of explanations is limited by the resolution of the dataset, which in the case of CIFAR-10 may reduce interpretability.

3. **Utility**:
   - Helps debug the model by identifying which features or regions contribute positively or negatively to predictions.
   - Enhances trust in model predictions by providing intuitive, visual insights into decision-making.
   - Facilitates comparisons between model predictions and ground truth, helping to identify potential biases or errors in the model.

#### **WHY**
SHAP values are used to interpret the predictions of a learning model by quantifying the contribution of each input pixel to the prediction. This provides transparency into model behavior, enabling developers and users to understand the "reasoning" behind specific decisions. SHAP helps improve trust in models, identify biases, debug errors, and ensure that the model focuses on relevant input features.

#### **WHO**
- **Model Developers**: To validate and debug models by understanding feature importance.
- **Model Users**: To assess whether predictions are reliable and based on meaningful patterns.
- **Non-experts**: To gain confidence in the model's decisions through visual and intuitive explanations.

#### **WHAT**
- **Visualized Data**: Pixel-level contributions to the model’s predictions.
- **Focus**: Highlights specific image regions or features that influence the prediction, providing a detailed breakdown of model decision-making.

#### **HOW**
- **Process**:
  - SHAP values are computed using a game-theoretic approach to measure the marginal contribution of each feature.
  - For images, regions are masked (e.g., blurred) to observe how predictions change.
  - SHAP values are visualized as heatmaps, with colors indicating positive or negative contributions to the model's confidence in its predictions.
- **Tools Used**: PyTorch for predictions, SHAP library for computing values, and CIFAR-10 dataset for image classification.

#### **WHEN**
- **During Training**: To understand how the model learns from training data and identify potential biases or errors.
- **After Training**: To interpret predictions, improve transparency, and debug issues in real-world use cases.

#### **WHERE**
- **Applications**:
  - Research settings to analyze the inner workings of neural networks.
  - Practical domains like medical imaging, where pixel-level explanations can validate predictions (e.g., identifying tumor regions).
  - For CIFAR-10: To provide detailed insights into simple image classifiers and demonstrate the effectiveness of SHAP in educational and experimental settings.

#### **Conclusion**
SHAP values excel in providing detailed, pixel-level insights into model behavior, offering a more nuanced explanation compared to gradient-based methods like saliency maps. While computationally intensive, SHAP delivers robust and interpretable results that are useful for debugging, improving model reliability, and building trust in predictions.

### Model Explanation Technique 3: Invertible Concept-based Explanations

Invertible Concept-based Explanations is an unsupervised approach for discovering concepts 
(converesely to TCAV which requires a prior dataset of concepts).
It is designed for explaining an already trained CNN model, making it a model-specific, post-hoc XAI method. 
In addition, explanations are local and global, as concepts represent overall structures 
learned by the CNN that are evaluated on single instnaces.
The technique is based on non-negative matrix factorization(NMF). A ReLU-activated convolutional layer is 
taken because the non-negativity provided by the ReLU function is a prerequisite for applying NMF. The layer is 
collapsed into two dimensions and serves as input for NMF which results in two matrices:
feature score S can be understood as an indicator of the presence of a concept while the feature direction 
P defines how the concepts look like and represent the CAVs.

$$A \in \mathbb{R}^{(\text{n, h, w, c})} \rightarrow V \in \mathbb{R}^{(\text{n x h x w, c})}$$

$$V = SP + U \hspace{1cm} S\in \mathbb{R}^{(\text{n x h x w, c'})} \hspace{1cm} P\in \mathbb{R}^{(c', c)}$$

These extracted concepts can then be used for Testing Concept Activation Vectors to see
which concepts activate what image class the most. Moreover, feature scores S can be upscaled 
to visualize the presence of a specific concept in an image. 

To obtain higher level concepts, the last activation layer is chosen as the target layer 
to calculate CAVs. 

[More details of this XAI techniques can be found here.](https://arxiv.org/abs/2006.15417)

#### **Interpretation of Results**

1. **Key Observations**:
   - Invertible Concept-based Explanations is able to extract (some) meaningful concepts that our CNN has learned. With this, it becomes clear why the model performs better for the means of transportation: their concepts are much more nuanced and meaningful as other concepts. E.g. for vehicles, the concrete floor or wheels are correctly identified. For animal classes, the focus is more on basic colors.
   - The method allowed us to explore the learned concepts in an unbiased way which is advantageous when the true underlying concepts are unknown.
2. **Limitations**:
   - Due to the hyperparameter selection of the number of concepts, we restrict our concept discovery a lot. Because the true underlying number is unknown, we are not able to recover all concepts that the model has learned. Analysis of 10 concepts might have been too little concepts or too many and thus, we cannot fully understand the model.
   - In addition, the lack of concept labels makes it sometimes very difficult to understand the extracted concepts (as seen above). Plotting the highest concept presence and lowest concept presence images helps to identify what the might (not) be. However, this is not an exact determination.
   - Moreover, to extract meaningful concepts, it is important that the model has acutally learned to classify samples well. In our case, we can see that the concepts are not specific and clear for classes, where the CNN has a lower accuracy.
3. **Utility**:
   - The method helps us to understand how a model learns to discrimiate between classes. Sometimes, it only looks at color patterns (Concept 6) which explains a lower accuracy for animal classes as this is a redunant feature.
   - With this, we know what parts the CNN is lacking and could provide specific training examples that help to improve model performance for specific classes. In addition, we also see if there is a potential bias in the dataset the model learns without understanding the class itself.
3. **Answers to the Research Questions**:
   - What parts of an image are important to classify an image? The CNN looks for some classes at specific parts (for vehicles), for other classes, the colors are more important. In addition, the CNN not only looks at the objects itself but also uses their surroundings for making a prediction.
   - What patterns/structures has our model learned? It focuses a lot on colors, and the combination of colors. Besides, locations and shapes are also relevant (e.g. Concept 2 describing wheels)
   - What difficulties does the CNN have when correctly predicting specific images/classes? The CNN had problems for classes where it has not learned discriminative concepts and simply relies on color patterns to make a prediction.
   
#### **WHY**
The purpose of Invertible Concept-based explanations (ICE) is to make a black-box CNN interpretable. 
In addition, it helps to debug the model as it can be checked whether the learned concepts
are reasonable for predicting a certain class. 
Finally, this is a human-friendly XAI technique, meaning that concepts follow the notion of 
human-friendly concepts and are thus intuitive for non-expert users.

#### **WHO**
ICE is not targeted for a specific audience. However, it is especially suitable for non-experts
due to the intuitive understanding of concepts. 

#### **WHAT**
- **Visualized Data**: Activation maps are represented as feature scores and concept activation
 vectors by applying NMF on an activation layer activated with an image batch. Thus, 
it can be said that the learned model parameters (convolutional filters) are explained by 
extracting hidden concepts in them.
- **Focus**: Highlighting regions in an image where a concept is strongly present, 
and plotting TCAV scores that display the fraction of images of a class in which 
a specific concept is present.

#### **HOW**
- **Process**:
  - The visualization process (extracting feature scores S and CAVs P is described above in more detail)
- **Tools Used**: PyTorch, CIFAR-10 dataset, and visualization libraries like Matplotlib.
- TCAV scores are displayed as bar charts, one for each concept representing the fraction
of images of a specific class that have this concept present.
- Feature Scores are visualized by laying a heatmap over the image to highlight the regions
where a specific concept is strongly present.

#### **WHEN**
- **After Training**: This method is only applicable after training a CNN. In addition, to obtain
meaningful results, the underlying assumption is that the model performs somewhat well, 
otherwise it might not have learned promising concepts.


#### **WHERE**
- **Applications**:
  - ICE has been applied to in the medical domain for skin tumor classification [(Kim et al. 2018)](https://arxiv.org/abs/1711.11279)
  or to analyze a model's concepts for diabetic retinopathy prediction [(Lucieri et al. 2020)](https://ieeexplore.ieee.org/document/9206946)
#### **Conclusion**
  Invertible Concept-based Explanations provide human-friendly concepts. The advantage compared to TCAV
is the unsupervised approach of finding concepts. With this, no prior assumptions about concepts are made. 
However a big drawback is specifying the number of concepts to extract. There have been attempts to 
automate this using the cosine similarty. Nevertheless, for this dataset, we could not find a successful 
hyperparameter that is promising. 
Moreover, our extracted concepts have no meaning. Thus, we still need an approach for annotating them in 
order to make them truly explainable.


### Model Explanation Technique 4: Instance Flow
InstanceFlow is a visualization tool that allows users to analyze the learning behavior of classifiers over time on the instance-level. It creates a Sankey diagram that visualizes the flow of instances throughout epochs and a tabular view where interesting instances can be located. This makes it possible to evaluate model performance on a class level over time and then, if required, zoom in to look at individaul instances and their classification over time.

#### **Interpretation of Results**

1. **Key Observations**:
   - We can clearly see the learning behavior of the model over time
   - We could visualize over all classes and epochs. However, this will lead to a messy end result that is difficult to interpret and hard to work with. 
   - Starting with a broad evaluation of the model and then iteratively refining it to narrow down why it behaves in a certain way is a better approach.
   - The method is effective in getting key insight regarding the classifications of even individual instances.
2. **Limitations**:
   - As already mentioned above. If we have a multiclass classification problem with lots of classes over many epochs, visualizing everything at once can lead to a messy end result that is difficult to interpret and hard to work with.
   - To visualize the flow of an instance, we must evaluate the model on it after each epoch. For a new instance, we would have to either retrain the model while evaluating it or save the model after each epoch and then evaluate on these saved instances.
3. **Utility**:
   - Provides a quick visual check of model behavior during training
   - But it can also be used to debug it on a class or even an instant level.

#### **WHY**
- InstanceFlow allows for a full analysis of a classification model’s learning behavior.
- Temporal performance can be viewed on a class and instance level.
- This can give us, for example, insight into why a model misclassifies a specific instance that we are interested in.

#### **WHO**
- **Model Developers**: To validate and debug models on an instance-level.
- **Model Users**: To get insight into why a specific misclassification might have occurred
- **Non-experts**: To get insight into the behavior of the model training process over time

#### **WHAT**
- **Visualized Data**: Classifications of the model on an instance level during training over time
- **Focus**: The flow and development of correct/incorrect classifications over time

#### **HOW**
- **Process**: Store model predictions on the dataset after each epoch during training and later visualize the flow (how the model predictions change) over time.
- **Tools Used**: PyTorch, CIFAR-10 dataset, the prototype tool accompanying the paper that introduced this technique: instanceflow.pueh.xyz

#### **WHEN**
- **During Training**: To debug the model and validate the choice of dataset and model hyperparameters.
- **After Training**: To get a peek into how the model developed during training after the fact.

#### **WHERE**
- **Applications**: This technique is model agnostic and can be useful to gain insight into the development of any classification model during training.

#### **Conclusion**
Instance flow provides a framework that gives a greater understanding of the temporal progression of predicted class distributions.
It can be used to debug misclassifications and to explore temporal patterns in classification accuracy.
Model performance over time during training can be evaluated on a class level, and the flow of the predictions for individual instances can give new insights.  

## Report Conclusion

In general, the combination of the 4 methods helped a lot in getting a deeper understanding of our model and thus answering our research questions: 

### Key Takeaways:
1. **Instance Flow**:  
   - Using Instance Flow as a first analysis gave us an intuition of where our model performs well and where it struggles.  
   - Identifying specific difficulties helped us explore ambiguous classes or instances in more detail with other XAI methods.  
   - In our case, we found that cat images are particularly ambiguous, often being misclassified as dogs or frogs.

2. **Input Feature Analysis**:  
   - Analyzing the input features revealed that the model focuses on the actual objects in the images and not on specific object parts or backgrounds.  
   - This was consistent across correctly and incorrectly predicted images, hinting that the model uses general patterns (e.g., colors) to make predictions.  
   - This also suggests that some images might be too ambiguous to be classified correctly, regardless of model improvements.
   - SHAP visualizations added detailed pixel-level explanations, highlighting the specific contributions of each feature (pixel) to the model's predictions.  
   - This method confirmed that the model primarily focuses on object regions rather than irrelevant background areas, regardless of prediction correctness.  
   - SHAP also revealed that the model relies heavily on general patterns, such as prominent color contrasts, which might explain its struggles with ambiguous classes like cats and dogs.  
   - By comparing SHAP explanations across instances, we could identify consistent features the model used for certain classes, providing actionable insights for targeted improvement.

3. **Concept-Based Analysis**:  
   - Adding a concept-based technique provided more specific insights into what the model has learned.  
   - We discovered that for high-performing classes, the model has learned fine-grained concepts (e.g., colors, shapes, and spatial combinations), indicating better learning behavior.  
   - Conversely, for lower-accuracy classes, the concepts were very basic (e.g., simple color patterns), leading to frequent confusion and misclassification.
   


### Moving Forward:
- Based on our performed analyses, we can efficiently improve the model by continuing the training process with newly added images for classes that were incorrectly predicted.  
- Using SHAP, we can further validate if the new training data influences the model to focus on more fine-grained features instead of broad patterns.  
- Post-training, re-evaluating the model with these methods would allow us to confirm improved learning behavior and accuracy, particularly for the challenging classes.

## Final Submission
The submission is done with this repository. Make to push your code until the deadline.

The repository has to include the implementations of the picked approaches and the filled out report in this README.

* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Save your final executed notebook(s) as html (File > Download as > HTML) and add them to your repository.

## Development Environment

Checkout this repo and change into the folder:
```
git clone https://github.com/jku-icg-classroom/xai_model_explanation_2024-<GROUP_NAME>.git
cd xai_model_explanation_2024-<GROUP_NAME>
```

Load the conda environment from the shared `environment.yml` file:
```
conda env create -f environment.yml
conda activate xai_model_explanation
```

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:
```
jupyter lab
```

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.


## Report

### Model & Data

* Which model are you going to explain? What does it do? On which data is it used?
* From where did you get the model and the data used?
* Describe the model.

### Explainability Approaches
Find explainability approaches for the selected models. Repeat the section below for each approach to describe it.

#### Approach *i*

* Briefly summarize the approach. 
* Categorize this explainability approach according to the criteria by Hohman et al.
* Interpret the results here. How does it help to explain your model?

### Summary of Approaches
Write a brief summary reflecting on all approaches.

### Presentation Video Link
Provide the link to your recorded video of the presentation. This could, for instance, be a Google Drive link. Make sure it is actually sharable and test it, e.g., by trying to access it from an incognito browser tab.
