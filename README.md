# Financial Sentiment Analysis 📈💼

A comprehensive machine learning project for analyzing sentiment in financial text data. This project implements both traditional machine learning algorithms and deep learning models to classify financial statements and news into positive, negative, or neutral sentiments.

## 🎯 Project Overview

Financial sentiment analysis is crucial for understanding market emotions and making informed investment decisions. This project analyzes financial text data to determine the sentiment expressed, which can be valuable for:

- **Investment Strategy**: Understanding market sentiment to guide investment decisions
- **Risk Assessment**: Identifying negative sentiment trends that might indicate potential risks
- **Market Research**: Analyzing public opinion about financial instruments or companies
- **Automated Trading**: Incorporating sentiment scores into algorithmic trading strategies

## ✨ Features

- **Multiple ML Models**: Implementation of various algorithms including Naive Bayes, Random Forest, SVM, Decision Trees, and Logistic Regression
- **Deep Learning Models**: Advanced neural networks including RNN and LSTM for improved accuracy
- **Text Preprocessing**: Comprehensive text cleaning with NLTK including tokenization, stemming, lemmatization, and stop word removal
- **Data Visualization**: Clear charts and graphs showing sentiment distribution and model performance
- **Model Comparison**: Performance evaluation across different algorithms to find the best approach
- **High Accuracy**: Achieves ~87% test accuracy with LSTM model

## 📊 Dataset Information

The project uses financial text data with the following characteristics:

- **Total Samples**: ~4,845 financial statements/news articles
- **Sentiment Distribution**:
  - Neutral: 2,878 samples (59.4%)
  - Positive: 1,363 samples (28.1%)
  - Negative: 604 samples (12.5%)
- **Data Format**: CSV file with text and corresponding sentiment labels
- **Data Source**: Financial news articles and company statements

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.x**: Main programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization

### Natural Language Processing
- **NLTK**: Text preprocessing and tokenization
- **Scikit-learn**: Traditional machine learning algorithms
- **WordNetLemmatizer**: Text lemmatization
- **PorterStemmer**: Text stemming

### Machine Learning & Deep Learning
- **Scikit-learn**: Traditional ML models (Naive Bayes, SVM, Random Forest, etc.)
- **TensorFlow/Keras**: Deep learning framework for RNN and LSTM models
- **LabelEncoder**: Target encoding
- **CountVectorizer**: Text vectorization

## 📁 Project Structure

```
Financial_Sentiment_Analysis/
│
├── README.md                    # Project documentation
├── Preprocess[1].ipynb         # Main Jupyter notebook with complete analysis
└── all-data.csv               # Dataset (not included in repository)
```

## 🚀 Installation & Setup

### Prerequisites
Make sure you have Python 3.7+ installed on your system.

### Required Dependencies

```bash
pip install pandas numpy matplotlib
pip install nltk scikit-learn
pip install tensorflow keras
pip install jupyter notebook
```

### NLTK Data Downloads
After installing NLTK, run the following in Python to download required data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 💻 Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/pawanchinnu17/Financial_Sentiment_Analysis.git
   cd Financial_Sentiment_Analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt  # If requirements.txt exists
   # Or install manually as listed above
   ```

3. **Prepare Dataset**
   - Place your financial text dataset as `all-data.csv` in the project directory
   - Ensure the CSV has columns: 'Sentiment' and 'text'

4. **Run the Analysis**
   ```bash
   jupyter notebook Preprocess[1].ipynb
   ```

5. **Execute Notebook Cells**
   - Run cells sequentially to perform the complete analysis
   - Modify parameters as needed for your specific dataset

## 📈 Model Performance

### Traditional Machine Learning Models
- **Multinomial Naive Bayes**: Baseline performance for text classification
- **Random Forest**: Ensemble method with good generalization
- **Support Vector Machine (SVM)**: Effective for high-dimensional text data
- **Decision Tree**: Interpretable model for understanding decision patterns
- **Logistic Regression**: Linear model with probabilistic output

### Deep Learning Models
- **Simple RNN**: Basic recurrent neural network for sequence processing
- **LSTM (Long Short-Term Memory)**: 
  - **Test Accuracy**: ~87%
  - **Best performing model** in the project
  - Excellent for capturing long-term dependencies in text

### Text Preprocessing Pipeline
1. **Lowercase Conversion**: Standardize text case
2. **Punctuation Removal**: Clean special characters
3. **Stop Word Removal**: Remove common English words
4. **Tokenization**: Split text into individual words
5. **Lemmatization/Stemming**: Reduce words to root forms

## 🔮 Future Improvements

### Model Enhancements
- [ ] **BERT Integration**: Implement transformer-based models for better context understanding
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy
- [ ] **Hyperparameter Tuning**: Optimize model parameters using GridSearch or RandomSearch
- [ ] **Cross-Validation**: Implement k-fold cross-validation for robust evaluation

### Data & Features
- [ ] **Larger Dataset**: Incorporate more diverse financial text sources
- [ ] **Feature Engineering**: Add TF-IDF, n-grams, and sentiment lexicon features
- [ ] **Real-time Analysis**: Process live financial news feeds
- [ ] **Multi-class Granularity**: Extend beyond positive/negative/neutral to more nuanced sentiments

### Technical Improvements
- [ ] **Model Deployment**: Create REST API for sentiment prediction
- [ ] **Performance Optimization**: Implement caching and batch processing
- [ ] **Visualization Dashboard**: Interactive web interface for results
- [ ] **A/B Testing Framework**: Compare model performance in production

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make Your Changes**
   - Add new models or improve existing ones
   - Enhance data preprocessing techniques
   - Improve documentation or add examples
4. **Commit Your Changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
5. **Push to the Branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Contribution Ideas
- Add new machine learning models
- Improve text preprocessing techniques
- Create visualization tools
- Add comprehensive unit tests
- Improve documentation and examples

## 📄 License

This project is available under the MIT License. See the LICENSE file for more details.

## 👤 Author

**Pawan Chinnu**
- GitHub: [@pawanchinnu17](https://github.com/pawanchinnu17)

## 🙏 Acknowledgments

- Financial dataset providers for making data available for research
- NLTK and scikit-learn communities for excellent NLP and ML tools
- TensorFlow team for powerful deep learning capabilities
- Open source community for inspiration and resources

## 📞 Support

If you have any questions or need help with the project:

1. **Issues**: Open an issue on GitHub for bugs or feature requests
2. **Discussions**: Use GitHub Discussions for general questions
3. **Documentation**: Check this README and inline code comments

---

⭐ **Star this repository if you found it helpful!** ⭐