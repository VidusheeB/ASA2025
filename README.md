# 🏛️ Political Leaning Classifier

A fine-tuned ChatGPT model for political leaning classification based on TF-IDF scores of political terms.

## 🚀 Features

- **Fine-tuned ChatGPT Model**: Uses OpenAI's fine-tuning API for political leaning classification
- **Interactive Web UI**: Beautiful Streamlit interface for easy interaction
- **K-Fold Cross-Validation**: Comprehensive evaluation with confusion matrices and metrics
- **Batch Processing**: Upload CSV files for bulk analysis
- **Real-time Analysis**: Single article classification with confidence scores
- **Detailed Metrics**: Accuracy, precision, recall, F1-score, specificity, and sensitivity

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Fine-tuned model name (required)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ASA2025
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export FINE_TUNED_MODEL_NAME="ft:gpt-4.1:your-model-id"  # Required
   ```

## 🎯 Usage

### Web Interface

Run the Streamlit web application:

```bash
streamlit run app.py
```

The web interface provides:
- **Single Prediction**: Input TF-IDF scores for individual articles
- **Batch Analysis**: Upload CSV files for bulk classification
- **K-Fold Evaluation**: Run cross-validation on training data
- **Model Information**: Configuration status and model details

### Command Line K-Fold Evaluation

Run comprehensive evaluation from command line:

```bash
python run_kfold_evaluation.py
```

This will:
- Load training data from `political_leaning_training.jsonl`
- Perform K-fold cross-validation
- Generate confusion matrices and performance plots
- Save detailed results to CSV files

## 📊 Model Architecture

### Input Format
The model expects TF-IDF scores for 24 political terms:
- Scrutiny, hostile attitudes, national security, political activism
- privilege, authoritarianism, suppression, disinformation
- unlawful, vetting, inhumane conditions, rights violations
- deportation, denaturalization, fraud, abuse, exploitation
- fear, misinformation, cruelty, unconstitutional, terror network, chaos

### Output
- **Prediction**: "left-leaning" or "right-leaning"
- **Confidence**: Score between 0.0 and 1.0

### Training Data Format
The model was trained on JSONL format with messages:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a political leaning classifier..."
    },
    {
      "role": "user", 
      "content": "TF-IDF scores: {...}"
    },
    {
      "role": "assistant",
      "content": "This article is left-leaning."
    }
  ]
}
```

## 📈 Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (True negatives + False positives)
- **Sensitivity**: True positives / (True positives + False negatives)

## 📁 File Structure

```
ASA2025/
├── app.py                          # Streamlit web application
├── fine_tuned_classifier.py        # Core classifier class
├── kfold_evaluator.py             # K-fold evaluation system
├── run_kfold_evaluation.py        # Command line evaluation script
├── political_leaning_training.jsonl # Training data
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `FINE_TUNED_MODEL_NAME` | Fine-tuned model name | Yes |

### Model Configuration

The system requires a fine-tuned model:

- **Fine-tuned Model**: Set `FINE_TUNED_MODEL_NAME` to your model ID
- **No Fallback**: The system will fail if no fine-tuned model is specified

## 📊 Output Files

When running K-fold evaluation, the system generates:

- `kfold_results.csv`: Detailed fold-wise and overall metrics
- `kfold_confusion_matrix.png`: Confusion matrix visualization
- `kfold_performance.png`: Fold-wise performance comparison

## 🎨 Web Interface Features

### Single Prediction
- Interactive TF-IDF score input
- Real-time classification with confidence scores
- Visual representation of key political terms
- Sample data loading for testing

### Batch Analysis
- CSV file upload for bulk processing
- Progress tracking during classification
- Summary statistics and downloadable results
- Error handling and validation

### K-Fold Evaluation
- Configurable number of folds (3, 5, 10)
- Real-time progress tracking
- Interactive visualizations with Plotly
- Comprehensive metrics display

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   Error: OpenAI API key not found
   ```
   Solution: Set the `OPENAI_API_KEY` environment variable

2. **Model Not Found**:
   ```
   Error: Fine-tuned model name not provided
   ```
   Solution: Set `FINE_TUNED_MODEL_NAME` - this is required for the system to work

3. **Training Data Missing**:
   ```
   Error: Training data file not found
   ```
   Solution: Ensure `political_leaning_training.jsonl` is in the current directory

### Performance Tips

- Use batch processing for large datasets to minimize API calls
- Set appropriate temperature (0.1) for consistent results
- Monitor API rate limits during evaluation
- Use the command line script for automated evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the fine-tuning API
- Streamlit for the web framework
- The political science community for research insights

---

**Note**: This system is designed for research and educational purposes. Always ensure compliance with OpenAI's usage policies and ethical guidelines when analyzing political content.
