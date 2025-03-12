# Comparative Analysis of Machine Learning Models for Breast Cancer Detection Using Nuclei Morphology Features


**Abstract**
Breast cancer remains one of the most prevalent and life-threatening cancers worldwide, with early and accurate diagnosis playing a crucial role in improving patient outcomes. Traditionally, pathologists rely on microscopic examination of cell nuclei to assess malignancy, as abnormal nuclei size and shape are strong indicators of cancerous transformation. However, manual assessment is time-consuming, subject to variability, and increasingly difficult given the growing demand for pathology services. The integration of digital pathology and machine learning offers a promising solution by enabling faster, more consistent, and automated classification of breast cancer.

In this study, we develop a machine-learning model that predicts breast cancer based on nuclei size and shape features extracted from histopathological images. We implement and compare a Random Forest classifier and a multi-layer perceptron (MLP) deep learning model. Their performance is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. Both models achieve high classification accuracy, with the MLP showing slightly improved recall, making it better suited for identifying malignant cases.

Additionally, we analyse feature importance in the Random Forest model and explore the potential benefits of combining both models through an ensemble approach. Visualizations such as ROC curves and confusion matrices further illustrate the comparative performance of these models.
This study contributes to the growing field of AI-assisted diagnostics, demonstrating how computational tools can enhance early breast cancer detection. Future work will focus on integrating convolutional neural networks (CNNs) for medical image analysis and developing ensemble models to further improve classification performance.



**1. Introduction**

Breast cancer remains a leading cause of cancer-related mortality worldwide, with early detection playing a critical role in improving patient survival rates. Pathologists typically assess breast tissue samples under a microscope, examining the size, shape, and structure of cell nuclei to determine malignancy. Malignant cells often exhibit irregularly enlarged nuclei with high variability in shape and texture. However, manual assessment is time-consuming, subject to inter-observer variability, and challenging when pathologists face large volumes of samples.

The rise of digital pathology and whole-slide imaging (WSI) has created opportunities for machine learning to assist in faster, more accurate diagnoses. Machine learning can automate nuclei feature extraction and classification, improving diagnostic accuracy and reducing the workload for clinicians. These models can process vast amounts of data in real-time, flagging suspicious cases for further review and accelerating cancer triage.



 ![image](https://github.com/user-attachments/assets/e36c6215-bde1-4e57-928c-24d138f5c6cf)

Figure 1. Examples of benign (left) and malignant (right) breast specimens stained with haematoxylin and eosin, at different magnifications. (Image source: Levenson et al., 2025).

**Primary Objectives:**

•	Analyse the relationship between nuclei morphology and malignancy, validating its role as a diagnostic feature.

•	Develop predictive models that classify breast cancer using nuclei size and shape features, offering a computational alternative to traditional pathology assessments.

•	Compare Random Forest (RF) and Multi-Layer Perceptron (MLP) models to determine their effectiveness in breast cancer prediction.

•	Explore feature importance and discuss the potential of an ensemble model combining RF and MLP.

By leveraging digital pathology and machine learning, this study aims to contribute to the advancement of AI-assisted diagnostics, where automated tools can enhance early breast cancer detection.



**2. Methodology**

**2.1 Dataset and Features**

The dataset used in this study is derived from the Wisconsin Breast Cancer Dataset, which contains nuclei size and shape measurements obtained from fine-needle aspiration biopsies. The dataset includes both benign and malignant cases, with features describing the structural characteristics of the cell nuclei.

Key features used for prediction include:

•	Mean Radius: The average size of the nucleus.

•	Mean Texture: The variation in cell structure across the sample.

•	Mean Perimeter: The boundary length of the nucleus.

•	Mean Area: The total area occupied by the nucleus.

•	Compactness: A measure of how tightly packed the cell structures are, which can indicate abnormal growth.


**2.2 Data Preprocessing**

Before training the machine learning models, the dataset was preprocessed to improve model accuracy and efficiency. 

The preprocessing steps included:

•	Handling Missing Data: Any missing values were removed or imputed using mean substitution.

•	Feature Scaling: Standardization was applied to ensure that all features were on the same scale, preventing bias toward larger numerical values.

•	Train-Test Split: The dataset was divided into training (80%) and testing (20%) sets to evaluate the model's performance.


**2.3 Machine Learning Model Implementation**

We compare two classification models:

**Random Forest Classifier**
Random Forest is an ensemble learning method that builds multiple decision trees and merges them to improve classification accuracy and control overfitting. It is well-suited for datasets with complex feature interactions.

**Multi-Layer Perceptron (MLP) Deep Learning Model**
MLP is a deep learning model consisting of:

•	Input Layer: Accepts standardized nuclei size and shape features.

•	Hidden Layers: Two fully connected layers with ReLU activation to capture complex relationships.

•	Output Layer: A single neuron with sigmoid activation for binary classification.


**2.4 Evaluation Metrics**

Models are assessed using:

•	Accuracy: Correctly classified cases.

•	Precision: Proportion of predicted malignant cases that are actually malignant.

•	Recall: Ability to correctly identify malignant cases.

•	F1-Score: Harmonic mean of precision and recall.

•	ROC-AUC: Measures classification performance across different thresholds.


**3. Results and Analysis**
**3.1 Model Performance Comparison**

<img width="388" alt="image" src="https://github.com/user-attachments/assets/95300080-e504-4207-b286-bcb6daa1e0fc" />

Both models perform similarly in accuracy and ROC-AUC, but the MLP demonstrates higher recall, making it more reliable for identifying malignant cases.



**3.2 Feature Importance Analysis**
Feature importance was analysed to determine which attributes had the most impact on classification.

Top 3 most important features:

1.	Mean Perimeter

2.	Mean Area

3.	Mean Radius

<img width="398" alt="image" src="https://github.com/user-attachments/assets/0711edf3-a906-44da-bbf0-7791927f2546" />

Figure 2. Feature important analysis of top five most important feature attributes.

**3.3 Data Visualisation**

ROC Curve Insights

Both models show excellent performance in distinguishing between malignant and benign cases, with AUC values of 0.96 (Random Forest) and 0.97 (MLP). MLP's higher AUC suggests it is slightly better at identifying malignant cases, particularly in more complex instances.  Both models are highly effective and could be used in complementary ways (e.g., in an ensemble model) to improve classification performance further.

<img width="284" alt="image" src="https://github.com/user-attachments/assets/28112622-df9c-4f0c-851f-ba6f3f579178" />

Figure 3. ROC curves for the Random Forest and MLP models. 


**4. Discussion**
The study shows that both the Random Forest and MLP models performed well in classifying breast cancer based on cell nuclei features. However, the MLP model demonstrated better recall, which is crucial for identifying true malignant cases and reducing the likelihood of false negatives. Given that early detection is critical in breast cancer treatment, the higher recall of MLP makes it an appealing choice for this task.

Random Forest, on the other hand, offers the advantage of interpretability. By analysing the feature importance, we gain valuable insights into which characteristics of the cell nuclei contribute most to the prediction of malignancy. This makes Random Forest useful not only for classification but also for understanding the biological factors that drive cancer development.

One possible future improvement is the combination of these two models in an ensemble approach. By leveraging the stability and feature analysis strength of Random Forest with the deep learning capabilities of MLP, we could create a more robust and accurate predictive tool.
Additionally, using Convolutional Neural Networks (CNNs) for image-based analysis could further enhance the model’s ability to recognize complex patterns in histopathological images.



**5. Conclusion and Future Work**
Future directions include:

•	Exploring CNNs for image-based classification.

•	Developing an ensemble model.

•	Deploying a real-world AI diagnostic tool.

This study highlights the potential of AI in early breast cancer detection, improving accuracy while reducing clinical workload.




