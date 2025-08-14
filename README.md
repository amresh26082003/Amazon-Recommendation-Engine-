# Amazon Apparel Recommendations Engine

A comprehensive, end-to-end recommendation engine for Amazon apparel products, leveraging state-of-the-art Natural Language Processing (NLP), Computer Vision, and Machine Learning techniques to provide product similarity and recommendation functionalities.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Modeling Approaches](#modeling-approaches)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Results & Visualization](#results--visualization)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project implements a multi-modal recommendation engine for Amazon apparel products. It uses product metadata, textual descriptions, and image features to recommend similar products using various algorithms, including Bag-of-Words, TF-IDF, IDF, Word2Vec, and CNN-based visual similarity.

---

## Features

- **Data Preprocessing:** Cleans and deduplicates product data, handles missing values, and preprocesses text.
- **Text-based Similarity:** Implements Bag-of-Words, TF-IDF, and IDF-based similarity on product titles.
- **Semantic Similarity:** Uses Word2Vec embeddings (average and IDF-weighted) for semantic similarity.
- **Visual Similarity:** Extracts deep features from product images using VGG16 CNN and computes visual similarity.
- **Brand & Color Weighting:** Incorporates brand and color information into similarity calculations.
- **Visualization:** Provides heatmaps and plots for similarity analysis and model comparison.

---

## Project Structure

```
Amazon-Apparel-Recommendations-Engine-main/
│
├── Amazon apparel Engine.ipynb      # Main Jupyter notebook with all code and analysis
├── LICENSE                         # Apache 2.0 License
├── README.md                       # Project documentation (this file)
├── pickels/                        # Preprocessed data files (pickle format)
├── images/                         # Downloaded product images
├── 16k_data_cnn_features.npy       # CNN features for images (download link in notebook)
├── 16k_data_cnn_feature_asins.npy  # ASINs corresponding to CNN features
└── word2vec_model                  # Trained Word2Vec model (download link in notebook)
```

---

## Data Preparation

- **Raw Data:** JSON file with Amazon apparel product metadata.
- **Preprocessing:** 
  - Remove entries with missing price or color.
  - Filter out products with short or duplicate titles.
  - Save cleaned data as pickle files for efficient loading.

---

## Modeling Approaches

1. **Bag-of-Words (BoW):**  
   Uses `CountVectorizer` to represent product titles and computes cosine similarity.

2. **TF-IDF:**  
   Uses `TfidfVectorizer` for weighted term frequency-inverse document frequency representation.

3. **IDF:**  
   Custom implementation to weigh words by their inverse document frequency.

4. **Word2Vec:**  
   - **Average:** Averages word embeddings for each title.
   - **IDF-Weighted:** Weights word embeddings by their IDF values.

5. **Visual Similarity (CNN):**  
   - Extracts features from images using VGG16 (Keras).
   - Computes Euclidean distances between image feature vectors.

6. **Brand & Color Weighted Similarity:**  
   - Combines text, brand, and color features for a holistic similarity score.

---

## How to Run

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/amresh26082003/Amazon-Recommendation-Engine-.git
   cd Amazon-Apparel-Recommendations-Engine-main
   ```

2. **Install Dependencies:**
   See [Dependencies](#dependencies) below.

3. **Prepare Data:**
   - Place your raw JSON data in the appropriate directory.
   - Run the notebook cells sequentially to preprocess data and generate features.

4. **Download Pretrained Models & Features:**
   - Download the provided pickle, Word2Vec, and CNN feature files from the links in the notebook.

5. **Run the Notebook:**
   - Open `Amazon apparel Engine.ipynb` in Jupyter Notebook or VS Code.
   - Execute cells as needed to explore and use the recommendation engine.

---

## Dependencies

- Python 3.6+
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk
- gensim
- plotly
- keras
- tensorflow
- PIL (Pillow)
- requests
- bs4 (BeautifulSoup4)

Install all dependencies using pip:
```sh
pip install -r requirements.txt
```
*(Create a `requirements.txt` file with the above packages for convenience.)*

---

## Results & Visualization

- **Similarity Heatmaps:** Visualize word overlap and semantic similarity between product titles.
- **Image Display:** Shows recommended product images alongside similarity scores.
- **Model Comparison:** Plots average Euclidean distances for each model to compare performance.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgements

- Amazon for the product metadata dataset.
- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for deep learning frameworks.
- [NLTK](https://www.nltk.org/) and