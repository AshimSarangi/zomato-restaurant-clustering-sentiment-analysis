# Zomato Restaurant Clustering & Sentiment Analysis

Segment Zomato restaurants (cost, cuisine, rating) and classify customer review sentiment.

**Stack:** Python, Pandas, NumPy, Scikit-learn, NLTK, spaCy, TextBlob, XGBoost, Matplotlib, Seaborn, Plotly

## Data
`data/` holds two CSVs:
- **Restaurant metadata** (105 restaurants): name, cost, cuisines, collections, timings
- **Reviews** (10,000 reviews): reviewer, review text, rating, reviewer metadata, time

## What the notebook does
1. **Cleaning:** fix cost dtype, handle missing/invalid ratings, split reviewer metadata into reviews/followers, extract year/month/hour.
2. **EDA:** costliest/cheapest restaurants, cuisine frequency, top reviewers, ratings, review activity by hour, word clouds.
3. **Text preprocessing:** NLTK stopwords, punctuation removal, spaCy lemmatization.
4. **Sentiment analysis:** TextBlob polarity/subjectivity to label reviews, then TF-IDF + Naive Bayes, Random Forest, XGBoost, SVM. LDA topic modelling on reviews.
5. **Clustering:** one-hot encode cuisines, merge average rating per restaurant, K-Means (elbow method, k=5) on raw data and on MinMax-scaled + PCA data.

## Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook notebooks/zomato_clustering_sentiment_analysis.ipynb
```

## Limitations
- Sentiment labels come from TextBlob polarity, so the classifiers learn to imitate TextBlob rather than true human labels. Treat the accuracy as agreement with TextBlob.
- Only 105 restaurants are clustered; cluster conclusions are indicative.

## Acknowledgements
Adapted from the capstone notebook by [tawadesharad](https://github.com/tawadesharad/Zomato-Restaurant-Clustering-And-Sentiment-Analysis-capstone-project-4). Dataset: Zomato Restaurant Names/Metadata and Reviews (Indian restaurants, Hyderabad).
