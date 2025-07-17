# Political Leaning Classifier

A machine learning system that classifies political leaning (left vs right) of articles based on TF-IDF features using logistic regression.

## Overview

This project provides:
- **Model Training**: Train logistic regression on TF-IDF features
- **Prediction**: Classify new articles as left or right-leaning
- **Web Interface**: Streamlit app for easy upload and prediction
- **PDF Support**: Upload PDF documents for analysis

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will:
- Load the `Scores.csv` data
- Train a logistic regression model
- Save the model to `models/political_classifier.pkl`
- Save feature importance to `models/political_classifier_importance.csv`

### 3. Make Predictions

#### Option A: Command Line
```bash
# Predict from text
python predict.py --text "Your article text here"

# Predict from file
python predict.py --file article.txt

# Interactive mode
python predict.py --interactive
```

#### Option B: Web Interface
```bash
streamlit run app.py
```
Then:
1. Upload a PDF or text file
2. Or paste text directly
3. Click "Predict Political Leaning"
4. View results and feature analysis

#### Option C: Python API
```python
from predict import predict_from_text

prediction, probability, features, analysis = predict_from_text("Your article text")
```

## File Structure

```
ASA2025/
├── Scores.csv                    # Training data (TF-IDF scores + labels)
├── train_model.py               # Model training script
├── predict.py                   # Prediction script
├── app.py                     # Web interface
├── test_prediction.py          # Test script
├── usage_example.py            # Usage examples
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── models/                     # Saved models (created after training)
    ├── political_classifier.pkl
    └── political_classifier_importance.csv
```

## Data Format

### Training Data (`Scores.csv`)
- **Label column**: Binary political leaning (0=left, 1=right)
- **Feature columns**: TF-IDF scores for vocabulary words
- **Features include**: immigration terms, legal terms, administrative terms

### Example Features
- Immigration: `asylum`, `border security`, `deportation plans`
- Legal: `constitutional rights`, `lawsuits`, `ACLU`
- Administrative: `Trump administration`, `executive order`

## Model Performance

With the current dataset (8 samples):
- **Accuracy**: 50% (limited by small dataset)
- **Top Features**: 
  - Right-leaning: `deportation plans`, `Trump administration`, `sanctuary policies`
  - Left-leaning: `racial profiling`, `immigration law`, `border security`

## Usage Examples

### Test Predictions
```bash
python test_prediction.py
```

### Web Interface Features
1. **Analysis Dashboard**: View model performance and feature importance
2. **Article Predictor**: Upload PDFs or paste text for prediction
3. **Feature Analysis**: See which terms influenced the prediction
4. **Visualization**: Charts showing prediction probabilities and feature contributions

### Supported File Types
- **PDF**: Upload PDF documents (text extraction)
- **TXT**: Plain text files
- **CSV**: Files with text in first column or 'text' column

## API Usage

```python
from predict import predict_from_text

# Make prediction
prediction, probability, features, analysis = predict_from_text(
    "The Trump administration has announced new deportation plans..."
)

# Interpret results
if prediction == 0:
    print("LEFT-LEANING")
else:
    print("RIGHT-LEANING")

print(f"Confidence: {max(probability):.3f}")
```

## Model Interpretation

### Positive Coefficients (Right-leaning indicators)
- Terms that predict right-leaning when present
- Examples: `deportation plans`, `Trump administration`, `sanctuary policies`

### Negative Coefficients (Left-leaning indicators)
- Terms that predict left-leaning when present
- Examples: `racial profiling`, `immigration law`, `constitutional rights`

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Run `python train_model.py` first

2. **"No significant features detected"**
   - Article may not contain relevant political vocabulary
   - Try longer articles with more political content

3. **Low confidence predictions**
   - Normal for small training dataset
   - Consider expanding training data

4. **PDF extraction issues**
   - Ensure PDF contains extractable text (not scanned images)
   - Try copying text manually if extraction fails

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning
- `matplotlib`: Plotting
- `seaborn`: Statistical visualizations
- `streamlit`: Web interface
- `PyPDF2`: PDF text extraction

## Next Steps

1. **Expand Dataset**: Add more training articles for better performance
2. **Feature Engineering**: Add more political vocabulary
3. **Model Tuning**: Experiment with different algorithms
4. **Validation**: Test on larger, more diverse datasets

## License

This project is for educational and research purposes.
