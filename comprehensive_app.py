import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="Comprehensive TF-IDF Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .best-metric {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .config-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class ComprehensiveAnalysisViewer:
    def __init__(self):
        self.results = None
        self.datasets_info = None
        self.load_results()
    
    def load_results(self):
        """Load analysis results"""
        try:
            results_path = 'comprehensive_results/analysis_results.pkl'
            datasets_path = 'comprehensive_results/datasets_info.pkl'
            
            if os.path.exists(results_path) and os.path.exists(datasets_path):
                with open(results_path, 'rb') as f:
                    self.results = pickle.load(f)
                with open(datasets_path, 'rb') as f:
                    self.datasets_info = pickle.load(f)
                return True
            else:
                st.error("Analysis results not found. Please run 'comprehensive_analysis.py' first.")
                return False
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return False
    
    def get_best_configurations(self):
        """Get best configurations based on different metrics"""
        if not self.results:
            return None, None, None
        
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_cv_accuracy = max(self.results.items(), key=lambda x: x[1]['cv_accuracy_mean'])
        lowest_cv_std = min(self.results.items(), key=lambda x: x[1]['cv_accuracy_std'])
        
        return best_accuracy, best_cv_accuracy, lowest_cv_std
    
    def display_overview(self):
        """Display overview section"""
        st.markdown('<h1 class="main-header">📊 Comprehensive TF-IDF Analysis Dashboard</h1>', unsafe_allow_html=True)
        
        if not self.results:
            st.warning("No analysis results found. Please run the comprehensive analysis first.")
            return
        
        # Get best configurations
        best_accuracy, best_cv_accuracy, lowest_cv_std = self.get_best_configurations()
        
        # Display best performance prominently
        st.markdown(f"## 🏆 **Best Performance: {best_accuracy[0]}**")
        st.markdown(f"### **Accuracy: {best_accuracy[1]['accuracy']:.3f} ({best_accuracy[1]['accuracy']*100:.1f}%)**")
        
        # Calculate and display sensitivity and specificity for best config
        cm = best_accuracy[1]['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        optimal_threshold = best_accuracy[1].get('optimal_threshold_used', 0.5)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sensitivity", f"{sensitivity:.3f}")
        with col2:
            st.metric("Specificity", f"{specificity:.3f}")
        with col3:
            st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
        with col4:
            st.metric("K-Fold", f"{best_accuracy[1]['k']}")
        
        st.markdown("---")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Configurations", 
                len(self.results),
                help="Number of different TF-IDF word counts × fold configurations analyzed"
            )
        
        with col2:
            st.metric(
                "Best Accuracy", 
                f"{best_accuracy[1]['accuracy']:.3f}",
                f"{best_accuracy[0]}",
                help=f"Best accuracy achieved by {best_accuracy[0]}"
            )
        
        with col3:
            st.metric(
                "Best CV Accuracy", 
                f"{best_cv_accuracy[1]['cv_accuracy_mean']:.3f}",
                f"{best_cv_accuracy[0]}",
                help=f"Best cross-validation accuracy achieved by {best_cv_accuracy[0]}"
            )
        
        with col4:
            st.metric(
                "Lowest CV Std", 
                f"{lowest_cv_std[1]['cv_accuracy_std']:.3f}",
                f"{lowest_cv_std[0]}",
                help=f"Lowest cross-validation standard deviation achieved by {lowest_cv_std[0]}"
            )
        
        # Configuration summary with averages and standard deviations
        st.subheader("📋 Configuration Summary")
        
        # Calculate averages and standard deviations by word count and set
        summary_data = []
        
        # Group by word count and set
        word_counts = [20, 25, 30]
        sets = ['Set1', 'Set2', 'Set3']
        
        for word_count in word_counts:
            for set_name in sets:
                # Filter configurations for this word count and set
                matching_configs = []
                for config_name, result in self.results.items():
                    if f'{set_name}_{word_count}w' in config_name:
                        matching_configs.append(result['accuracy'])
                
                if matching_configs:
                    avg_accuracy = np.mean(matching_configs)
                    std_accuracy = np.std(matching_configs)
                    summary_data.append({
                        'Word Count': word_count,
                        'Set': set_name,
                        'Average Accuracy': f"{avg_accuracy:.3f}",
                        'Std Dev': f"{std_accuracy:.3f}",
                        'Configurations': len(matching_configs)
                    })
        
        # Calculate overall averages by word count
        st.subheader("📊 Overall Word Count Performance")
        
        overall_summary = []
        for word_count in word_counts:
            # Get all configurations for this word count
            word_configs = []
            for config_name, result in self.results.items():
                if f'{word_count}w' in config_name:
                    word_configs.append(result['accuracy'])
            
            if word_configs:
                avg_accuracy = np.mean(word_configs)
                std_accuracy = np.std(word_configs)
                overall_summary.append({
                    'Word Count': word_count,
                    'Average Accuracy': f"{avg_accuracy:.3f}",
                    'Std Dev': f"{std_accuracy:.3f}",
                    'Total Configurations': len(word_configs)
                })
        
        # Display overall summary
        overall_df = pd.DataFrame(overall_summary)
        st.dataframe(overall_df, use_container_width=True)
        
        # Display detailed summary by set and word count
        st.subheader("📊 Detailed Performance by Set and Word Count")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Original detailed configuration summary (collapsible)
        with st.expander("📋 Detailed Configuration Summary (All Configurations)"):
            config_summary = []
            for config_name, result in self.results.items():
                config_summary.append({
                    'Configuration': config_name,
                    'Accuracy': result['accuracy'],
                    'CV Accuracy Mean': result['cv_accuracy_mean'],
                    'CV Accuracy Std': result['cv_accuracy_std']
                })
            
            detailed_df = pd.DataFrame(config_summary)
            detailed_df = detailed_df.sort_values('Accuracy', ascending=False)
            st.dataframe(detailed_df, use_container_width=True)
    
    def display_confusion_matrices(self):
        """Display confusion matrices"""
        # Get best configuration for headline
        best_config = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_result = self.results[best_config]
        
        st.subheader(f"🎯 Confusion Matrices (Best: {best_config} - {best_result['accuracy']:.3f} accuracy)")
        
        if not self.results:
            return
        
        # Create tabs for different view options
        tab1, tab2 = st.tabs(["📊 All Matrices", "🏆 Best Configurations"])
        
        with tab1:
            # Set navigation
            sets = ['Set1', 'Set2', 'Set3']
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⬅️ Previous Set", key="prev_set"):
                    if 'current_set_idx' not in st.session_state:
                        st.session_state.current_set_idx = 0
                    st.session_state.current_set_idx = (st.session_state.current_set_idx - 1) % len(sets)
            
            with col3:
                if 'current_set_idx' not in st.session_state:
                    st.session_state.current_set_idx = 0
                current_set = sets[st.session_state.current_set_idx]
                st.markdown(f"**Current Set: {current_set}**")
            
            with col5:
                if st.button("Next Set ➡️", key="next_set"):
                    if 'current_set_idx' not in st.session_state:
                        st.session_state.current_set_idx = 0
                    st.session_state.current_set_idx = (st.session_state.current_set_idx + 1) % len(sets)
            
            # Filter configurations for current set
            current_set_configs = {k: v for k, v in self.results.items() if current_set in k}
            
            if not current_set_configs:
                st.warning(f"No configurations found for {current_set}")
                return
            
            # Display confusion matrices for current set
            n_configs = len(current_set_configs)
            n_cols = 3
            n_rows = (n_configs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (config_name, result) in enumerate(current_set_configs.items()):
                row = idx // n_cols
                col = idx % n_cols
                
                cm = result['confusion_matrix']
                
                # Calculate sensitivity and specificity
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Left', 'Right'], 
                           yticklabels=['Left', 'Right'],
                           ax=axes[row, col])
                
                axes[row, col].set_title(f'{config_name}\nAcc: {result["accuracy"]:.3f}, Sens: {sensitivity:.3f}, Spec: {specificity:.3f}')
                axes[row, col].set_xlabel('Predicted')
                axes[row, col].set_ylabel('Actual')
            
            # Hide empty subplots
            for idx in range(n_configs, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            # Display only best configurations
            best_accuracy, best_cv_accuracy, lowest_cv_std = self.get_best_configurations()
            best_configs = [best_accuracy, best_cv_accuracy, lowest_cv_std]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for idx, (config_name, result) in enumerate(best_configs):
                cm = result['confusion_matrix']
                
                # Calculate sensitivity and specificity
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Left', 'Right'], 
                           yticklabels=['Left', 'Right'],
                           ax=axes[idx])
                
                axes[idx].set_title(f'{config_name}\nAcc: {result["accuracy"]:.3f}, Sens: {sensitivity:.3f}, Spec: {specificity:.3f}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    def display_performance_comparison(self):
        """Display performance comparison charts"""
        st.subheader("📈 Performance Comparison")
        
        if not self.results:
            return
        
        # Prepare data for plotting
        config_names = list(self.results.keys())
        metrics = ['accuracy']
        
        # Create comparison chart using plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'F1-Score', 'Precision', 'Recall'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for idx, metric in enumerate(metrics):
            values = [self.results[config][metric] for config in config_names]
            
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(
                go.Bar(
                    x=config_names,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=colors[idx],
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Performance Metrics Across All Configurations"
        )
        
        # Update x-axis labels
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=14)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Threshold analysis method removed as requested
    

    
    def display_detailed_analysis(self):
        """Display detailed analysis with 3 matrices at a time (one word count per set with all 3 folds)"""
        st.subheader("🔬 Detailed Analysis - 3 Matrices at a Time")
        
        if not self.results:
            return
        
        # Word count selector
        word_counts = [20, 25, 30]
        selected_word_count = st.selectbox(
            "Select Word Count:",
            word_counts,
            help="Choose a word count to view all 3 sets (Set 1, Set 2, Set 3) with all fold configurations"
        )
        
        # Get configurations for the selected word count
        configs_for_word_count = []
        for config_name in self.results.keys():
            if f"{selected_word_count}w" in config_name:
                configs_for_word_count.append(config_name)
        
        if configs_for_word_count:
            # Sort by set number and fold type
            configs_for_word_count.sort()
            
            # Display 3 matrices in a row
            st.subheader(f"Confusion Matrices for {selected_word_count} Words")
            
            # Create 3 columns for the matrices
            cols = st.columns(3)
            
            for i, config_name in enumerate(configs_for_word_count):
                if i < 3:  # Only show first 3 configurations
                    result = self.results[config_name]
                    
                    with cols[i]:
                        # Calculate sensitivity and specificity
                        cm = result['confusion_matrix']
                        tn, fp, fn, tp = cm.ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        # Get optimal threshold
                        optimal_threshold = result.get('optimal_threshold_used', 0.5)
                        
                        # Display metrics
                        st.metric("Accuracy", f"{result['accuracy']:.3f}")
                        st.metric("Sensitivity", f"{sensitivity:.3f}")
                        st.metric("Specificity", f"{specificity:.3f}")
                        st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
                        
                        # Display confusion matrix
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=['Left', 'Right'],
                                   yticklabels=['Left', 'Right'])
                        plt.title(f'{config_name}\nAcc: {result["accuracy"]:.3f}, Sens: {sensitivity:.3f}, Spec: {specificity:.3f}\nThreshold: {optimal_threshold:.3f}')
                        plt.ylabel('True')
                        plt.xlabel('Predicted')
                        st.pyplot(fig)
                        plt.close()
            
            # Display summary table for the selected word count
            st.subheader(f"Summary for {selected_word_count} Words")
            
            summary_data = []
            for config_name in configs_for_word_count:
                result = self.results[config_name]
                cm = result['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                optimal_threshold = result.get('optimal_threshold_used', 0.5)
                
                summary_data.append({
                    'Configuration': config_name,
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'Sensitivity': f"{sensitivity:.3f}",
                    'Specificity': f"{specificity:.3f}",
                    'Optimal Threshold': f"{optimal_threshold:.3f}",
                    'CV Mean': f"{result['cv_accuracy_mean']:.3f}",
                    'CV Std': f"{result['cv_accuracy_std']:.3f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Find best configuration for this word count
            best_config = max(configs_for_word_count, key=lambda x: self.results[x]['accuracy'])
            best_result = self.results[best_config]
            
            st.success(f"**Best Configuration for {selected_word_count} words:** {best_config} (Accuracy: {best_result['accuracy']:.3f})")
        else:
            st.warning(f"No configurations found for {selected_word_count} words.")
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
            
            if not text.strip():
                return None, "No text could be extracted. This PDF may contain only images or scanned content."
            
            text = text.strip()
            word_count = len(text.split())
            if word_count < 10:
                return None, f"Very little text extracted ({word_count} words). This PDF may be image-based."
            
            return text, f"Successfully extracted {word_count} words from {total_pages} pages."
            
        except Exception as e:
            return None, f"Error reading PDF: {str(e)}"
    
    def extract_features_from_text(self, text, dataset_name):
        """Extract TF-IDF features from text based on the selected dataset"""
        if not self.datasets_info or dataset_name not in self.datasets_info:
            return None
        
        # Load the dataset to get feature names
        dataset_info = self.datasets_info[dataset_name]
        feature_names = dataset_info['feature_names']
        
        # Simple word counting approach
        import re
        from collections import Counter
        
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        # Create feature vector based on vocabulary
        features = []
        for term in feature_names:
            # Check for exact matches and partial matches
            count = 0
            for word in word_counts:
                if term.lower() in word.lower() or word.lower() in term.lower():
                    count += word_counts[word]
            features.append(count)
        
        # Normalize by text length
        if len(words) > 0:
            features = [f / len(words) for f in features]
        
        return features
    
    def predict_article(self, text, threshold=0.5):
        """Predict political leaning for given text using Set 2 25-words model"""
        try:
            # Load the trained Set 2 25-words model
            model_path = 'models/set2_25w_model.pkl'
            if not os.path.exists(model_path):
                st.error("Set 2 25-words model not found. Please run 'train_set2_25w_model.py' first.")
                return None, None, None
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            optimal_threshold = model_data['optimal_threshold']
            
            # Use optimal threshold if none provided
            if threshold == 0.5:
                threshold = optimal_threshold
            
            # Extract features using the same vocabulary as the trained model
            features = self.extract_features_from_text_set2_25w(text, feature_names)
            if features is None:
                return None, None, None
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            probability = model.predict_proba(features_scaled)[0]
            prediction = 1 if probability[1] >= threshold else 0
            
            return prediction, probability, features
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None, None
    
    def extract_features_from_text_set2_25w(self, text, feature_names):
        """Extract TF-IDF features from text using Set 2 25-words vocabulary"""
        import re
        from collections import Counter
        
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        # Create feature vector based on the trained model's vocabulary
        features = []
        for term in feature_names:
            # Check for exact matches and partial matches
            count = 0
            for word in word_counts:
                if term.lower() in word.lower() or word.lower() in term.lower():
                    count += word_counts[word]
            features.append(count)
        
        # Normalize by text length
        if len(words) > 0:
            features = [f / len(words) for f in features]
        
        return features
    
    def display_article_predictor(self):
        """Display article prediction interface"""
        st.subheader("📝 Article Predictor")
        st.write("Upload a document or paste text to predict its political leaning using the Set 2 25-words model (best performing).")
        
        # Load optimal threshold from trained model
        optimal_threshold = 0.5
        try:
            model_path = 'models/set2_25w_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                optimal_threshold = model_data['optimal_threshold']
        except:
            pass
        
        # Use optimal threshold (no slider)
        threshold = optimal_threshold
        
        # Input method selection
        input_method = st.radio(
                "Choose input method:",
                ["📄 Upload PDF", "📄 Upload Text File", "✏️ Paste Text"]
            )
        
        article_text = ""
        
        if input_method == "📄 Upload PDF":
            uploaded_file = st.file_uploader(
                "Upload a PDF file",
                type=['pdf']
            )
            
            if uploaded_file is not None:
                st.info("📰 **News Article Mode**: Standard extraction works best for articles with typed text.")
                
                with st.spinner("Extracting text from PDF..."):
                    article_text, extraction_status = self.extract_text_from_pdf(uploaded_file)
                    if article_text:
                        st.success(f"✅ PDF text extracted successfully! ({extraction_status})")
                    else:
                        st.error(f"❌ Failed to extract text from PDF: {extraction_status}")
        
        elif input_method == "📄 Upload Text File":
            uploaded_file = st.file_uploader(
                "Upload a text file (.txt) or CSV file",
                type=['txt', 'csv']
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.txt'):
                        article_text = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            article_text = df['text'].iloc[0]
                        else:
                            article_text = df.iloc[0, 0]  # First column
                    st.success("✅ File loaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Paste Text
            article_text = st.text_area(
                "Paste your article text here:",
                height=200,
                placeholder="Enter the article text to analyze..."
            )
        
        if article_text:
            if st.button("🔍 Predict Political Leaning", type="primary"):
                with st.spinner("Analyzing article..."):
                    try:
                        # Make prediction using Set 2 25-words model
                        prediction, probability, features = self.predict_article(article_text, threshold)
                        
                        if prediction is not None:
                            st.subheader("🎯 Prediction Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if prediction == 0:
                                    st.success("**Prediction: LEFT-LEANING** 🟦")
                                else:
                                    st.success("**Prediction: RIGHT-LEANING** 🟥")
                            
                            with col2:
                                confidence = max(probability)
                                st.metric("Confidence", f"{confidence:.3f}")
                            
                            with col3:
                                st.metric("Model Used", "Set 2 25-words")
                            
                            # Show probabilities
                            st.subheader("📊 Prediction Probabilities")
                            prob_df = pd.DataFrame({
                                'Political Leaning': ['Left-leaning', 'Right-leaning'],
                                'Probability': [probability[0], probability[1]]
                            })
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            colors = ['blue', 'red']
                            ax.bar(prob_df['Political Leaning'], prob_df['Probability'], color=colors)
                            ax.set_ylabel('Probability')
                            ax.set_title('Prediction Probabilities')
                            ax.set_ylim(0, 1)
                            for i, v in enumerate(prob_df['Probability']):
                                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                            st.pyplot(fig)
                            plt.close()
                            
                            # Show feature analysis
                            st.subheader("🔍 Feature Analysis")
                            if features:
                                # Load feature names from trained model
                                try:
                                    model_path = 'models/set2_25w_model.pkl'
                                    if os.path.exists(model_path):
                                        with open(model_path, 'rb') as f:
                                            model_data = pickle.load(f)
                                        feature_names = model_data['feature_names']
                                        
                                        # Create feature analysis
                                        feature_analysis = []
                                        for i, (feature, score) in enumerate(zip(feature_names, features)):
                                            if score > 0:
                                                feature_analysis.append({
                                                    'feature': feature,
                                                    'score': score
                                                })
                                        
                                        if feature_analysis:
                                            st.write("**Features found in your article:**")
                                            for feature in feature_analysis[:10]:  # Show top 10
                                                st.write(f"• {feature['feature']}: {feature['score']:.4f}")
                                        else:
                                            st.info("No significant features were detected in your article.")
                                except:
                                    st.info("Feature analysis not available.")
                            
                            # Show model info
                            st.subheader("📋 Model Information")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Model:** Set 2 25-words")
                                st.write(f"**Threshold:** {threshold:.3f}")
                            
                            with col2:
                                # Load model performance info
                                try:
                                    model_path = 'models/set2_25w_model.pkl'
                                    if os.path.exists(model_path):
                                        with open(model_path, 'rb') as f:
                                            model_data = pickle.load(f)
                                        performance = model_data['performance']
                                        st.write(f"**Test Accuracy:** {performance['accuracy']:.3f}")
                                        st.write(f"**F1-Score:** {performance['f1_score']:.3f}")
                                except:
                                    st.write("**Model:** Set 2 25-words")
                                    st.write("**Best performing configuration**")
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
            
            else:
                st.info("👆 Please upload a file or paste text to analyze.")
        else:
            st.error("No datasets available. Please run the comprehensive analysis first.")

def main():
    """Main function"""
    viewer = ComprehensiveAnalysisViewer()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Overview", "Confusion Matrices", 
         "Detailed Analysis", "Article Predictor"]
    )
    
    # Display selected page
    if page == "Overview":
        viewer.display_overview()
    elif page == "Confusion Matrices":
        viewer.display_confusion_matrices()
    elif page == "Detailed Analysis":
        viewer.display_detailed_analysis()
    elif page == "Article Predictor":
        viewer.display_article_predictor()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About:**")
    st.sidebar.markdown("This dashboard displays comprehensive analysis results comparing different TF-IDF word counts (20, 25, 30) with various fold configurations (5, 10, 1 fold).")
    
    if not viewer.results:
        st.sidebar.warning("⚠️ Run comprehensive_analysis.py first to generate results.")

if __name__ == "__main__":
    main()
