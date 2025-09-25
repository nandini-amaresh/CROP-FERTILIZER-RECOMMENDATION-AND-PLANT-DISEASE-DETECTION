# Agriculture_AI
ML and DL based website which recommends the best crop to grow, fertilizers to use and the diseases caught by your crops.

CROP,FERTILIZER RECOMMENDATION AND PLANT DISEASE DETECTION

PROBLEM STATEMENT

Farmers face multiple challenges in optimizing agricultural productivity due to changes in rainfall and temperature and lack of information regarding crop selection, fertilizer use, and disease management. To address this, a comprehensive intelligent system is proposed that encompasses three key components: (a) a machine learning-based crop recommendation model that suggests the most suitable crop based on soil nutrients (N, P, K), pH, temperature, humidity, and rainfall to enhance yield and resource efficiency; (b) a fertilizer recommendation engine that analyzes nutrient deficiencies and crop type to provide balanced fertilizer suggestions, thereby improving soil health and productivity; and (c) a deep learning-powered plant disease detection system that identifies diseases from leaf images for early intervention and effective treatment, reducing crop losses and supporting sustainable farming practices.

DATA SET

The Crop recommendation dataset contains 2200 records with 8 features related to soil and weather conditions such as nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, and rainfall. The target variable Crop indicates the most suitable crop to grow under those conditions. It is ideal for building classification models to recommend crops based on environmental inputs.

The Fertilizer dataset contains 99 records and 9 columns related to agricultural parameters used for predicting suitable fertilizers. It includes numeric features like temperature, humidity, moisture, and nutrient levels (nitrogen, potassium, phosphorous), as well as categorical features such as soil type and crop type. The target variable is the "Fertilizer Name" recommended for each set of conditions. All entries are complete with no missing values

The "New Plant Diseases Dataset" on Kaggle is a comprehensive image dataset containing approximately 87,000 labeled images of plant leaves, both healthy and affected by various diseases. The dataset is organized into 38 classes, each representing a unique combination of crop type and disease condition, such as "Tomato - Late Blight" or "Apple - Scab." All images are RGB and typically feature centered leaves on a plain background, making them suitable for computer vision tasks. This dataset is commonly used in plant disease detection research and is ideal for training and evaluating deep learning models, particularly convolutional neural networks (CNNs)


SELECTED MODELS

CROP RECOMMENDATION

   Baseline Model: Logistic Regression

   Model chosen after evaluation :Random Forest Classifier and Gaussian NB

FERTILIZER RECOMMENDATION

   Model chosen after evaluation :Random Forest Classifier and SVC

PLANT DISEASE DETECTION

    Convolution Neural network

