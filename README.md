# CROP-FERTILIZER-RECOMMENDATION-AND-PLANT-DISEASE-DETECTION
An AI-based system that helps farmers by recommending the best crop to grow using soil and weather data, suggesting suitable fertilizers based on nutrient needs, and detecting plant diseases from leaf images with deep learning. This improves yield, soil health, and reduces crop losses.
PROBLEM STATEMENT
Farmers often struggle to maximize agricultural productivity due to unpredictable weather conditions such as rainfall and temperature variations, along with limited knowledge of crop selection, fertilizer usage, and disease management. To tackle these issues, we propose an integrated intelligent system that focuses on three major aspects:

Crop Recommendation – A machine learning model that suggests the most appropriate crop to cultivate based on soil nutrients (N, P, K), pH level, rainfall, temperature, and humidity. This ensures better yield and efficient use of resources.

Fertilizer Recommendation – A recommendation engine that evaluates nutrient imbalances and crop type to suggest suitable fertilizers, improving soil fertility and productivity.

Plant Disease Detection – A deep learning-based system that identifies crop diseases using leaf images, enabling early diagnosis and treatment, thus minimizing losses and promoting sustainable farming practices.

DATA SET

The Crop Recommendation dataset has 2200 entries and 8 features that capture soil and climatic conditions, including nitrogen, phosphorus, potassium, pH, rainfall, temperature, and humidity. The output label is the crop most suitable for those conditions. It is well-suited for classification tasks to predict the best crop for given environmental factors.

The Fertilizer dataset consists of 99 samples and 9 features. It includes numerical variables such as temperature, humidity, moisture, and soil nutrients (N, P, K), along with categorical variables like soil type and crop type. The target column specifies the recommended fertilizer. All records are complete without missing values.

The Plant Diseases dataset (sourced from Kaggle) contains about 87,000 RGB leaf images categorized into 38 classes, covering both healthy and diseased leaves across different crops. Each class represents a crop-disease pair (e.g., "Apple – Scab," "Tomato – Late Blight"). The dataset is widely used for plant disease recognition tasks and is suitable for training deep learning models, especially convolutional neural networks (CNNs).

SELECTED MODELS

Crop Recommendation: Logistic Regression was used as the baseline. After evaluation, Random Forest Classifier and Gaussian Naive Bayes were selected as the best-performing models.

Fertilizer Recommendation: Random Forest Classifier and Support Vector Classifier (SVC) were chosen after testing multiple approaches.

Plant Disease Detection: Deep learning models, particularly CNN-based architectures, were employed to achieve high accuracy in identifying diseases from leaf images.
