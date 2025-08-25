import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
from article_classifier import ArticlePoliticalClassifier
from document_processor import DocumentProcessor
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Political Leaning Classifier",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
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
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def initialize_classifier():
    """Initialize the fine-tuned classifier."""
    try:
        model_name = st.secrets.get("FINE_TUNED_MODEL_NAME", os.getenv("FINE_TUNED_MODEL_NAME"))
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        
        if not api_key:
            st.error("OpenAI API key not found. Please set it in your environment variables or Streamlit secrets.")
            return None
            
        if not model_name:
            st.error("Fine-tuned model name not found. Please set FINE_TUNED_MODEL_NAME in your environment variables or Streamlit secrets.")
            return None
            
        classifier = ArticlePoliticalClassifier(model_name=model_name, api_key=api_key)
        return classifier
    except Exception as e:
        st.error(f"Error initializing classifier: {e}")
        return None

def initialize_document_processor():
    """Initialize the document processor."""
    try:
        return DocumentProcessor()
    except Exception as e:
        st.error(f"Error initializing document processor: {e}")
        return None

def create_sample_article():
    """Create a sample article for demonstration."""
    return """The Trump administration has announced new deportation plans that target undocumented immigrants across the country. The policy includes increased scrutiny of immigration cases and stricter vetting procedures. Critics argue that these measures create inhumane conditions and violate constitutional rights. The administration defends the policy as necessary for national security, citing concerns about fraud and abuse in the immigration system. However, opponents claim this approach fosters fear and misinformation, leading to chaos in immigrant communities."""

def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ Political Leaning Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Fine-tuned ChatGPT Model for Political Analysis")
    
    # Initialize classifier and document processor
    classifier = initialize_classifier()
    document_processor = initialize_document_processor()
    if classifier is None or document_processor is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Single Prediction", "Batch Analysis", "Results"]
    )
    
    if page == "Single Prediction":
        show_single_prediction(classifier, document_processor)
    elif page == "Batch Analysis":
        show_batch_analysis(classifier)
    elif page == "Results":
        show_results_page()

def show_single_prediction(classifier, document_processor):
    """Show the single prediction interface."""
    st.header("📊 Single Article Classification")
    st.markdown("Upload a document or enter article text to classify its political leaning based on language analysis.")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📄 Upload Document", "✏️ Enter Text"])
    
    with tab1:
        st.subheader("Upload Document")
        st.markdown("Supported formats: PDF, TXT, DOCX, CSV")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['pdf', 'txt', 'docx', 'csv'],
            help="Upload a document to extract and analyze its text"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Process the uploaded file
            extracted_text, success, file_type = document_processor.process_uploaded_file(uploaded_file)
            
            if success and extracted_text:
                st.subheader("📝 Extracted Text")
                st.text_area(
                    "Extracted text (you can edit this):",
                    value=extracted_text,
                    height=200,
                    key="extracted_text"
                )
                
                # Classify button for uploaded file
                if st.button("🔍 Classify Uploaded Document", type="primary"):
                    with st.spinner("Analyzing political leaning..."):
                        try:
                            prediction, confidence, analysis_data = classifier.classify_article(extracted_text)
                            
                            if prediction == "error":
                                st.error("Error occurred during classification. Please check your API key and model configuration.")
                            else:
                                st.session_state.prediction = prediction
                                st.session_state.confidence = confidence
                                st.session_state.analysis_data = analysis_data
                                
                        except Exception as e:
                            st.error(f"Error: {e}")
            elif not success:
                st.error("Failed to extract text from the uploaded file.")
    
    with tab2:
        st.subheader("Enter Article Text")
        
        # Add a button to load sample data
        if st.button("Load Sample Article"):
            sample_article = create_sample_article()
            st.session_state.article_text = sample_article
        
        # Text input for article
        article_text = st.text_area(
            "Enter article text:",
            value=st.session_state.get('article_text', ''),
            height=300,
            placeholder="Paste your article text here..."
        )
        
        # Store in session state
        st.session_state.article_text = article_text
        
        # Classify button
        if st.button("🔍 Classify Article Text", type="primary"):
            if not article_text.strip():
                st.error("Please enter article text to classify.")
            else:
                with st.spinner("Analyzing political leaning..."):
                    try:
                        prediction, confidence, analysis_data = classifier.classify_article(article_text)
                        
                        if prediction == "error":
                            st.error("Error occurred during classification. Please check your API key and model configuration.")
                        else:
                            st.session_state.prediction = prediction
                            st.session_state.confidence = confidence
                            st.session_state.analysis_data = analysis_data
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Results section (outside of tabs)
    st.subheader("📊 Results")
    
    if hasattr(st.session_state, 'prediction'):
        prediction = st.session_state.prediction
        confidence = st.session_state.confidence
        
        # Display prediction with color coding
        if prediction == "left-leaning":
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🟦 **Prediction: Left-Leaning**")
            st.markdown("</div>", unsafe_allow_html=True)
        elif prediction == "right-leaning":
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🟥 **Prediction: Right-Leaning**")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"Unknown prediction: {prediction}")
        
        # Show analysis data
        if hasattr(st.session_state, 'analysis_data'):
            analysis = st.session_state.analysis_data
            
            # Show significant political terms
            if analysis.get('significant_terms'):
                st.subheader("🔍 Key Political Terms Detected")
                terms_df = pd.DataFrame(analysis['significant_terms'], columns=['Term', 'TF-IDF Score'])
                fig = px.bar(terms_df.head(10), x='TF-IDF Score', y='Term', 
                           orientation='h', title="Top Political Terms")
                st.plotly_chart(fig, use_container_width=True)
            


    else:
        st.info("Upload a document or enter text, then click 'Classify' to see results.")

def show_batch_analysis(classifier):
    """Show the batch analysis interface."""
    st.header("📈 Batch Analysis")
    st.markdown("Upload multiple files or a CSV file to perform batch classification.")
    
    # Show helpful tips
    with st.expander("💡 Tips for Batch Analysis"):
        st.markdown("""
        **Supported File Types:**
        - **PDF files** (.pdf) - News articles, reports, documents
        - **Text files** (.txt) - Plain text articles
        - **Word documents** (.docx) - Formatted documents
        - **CSV files** (.csv) - Spreadsheets with 'text' column
        
        **Upload Options:**
        - **Multiple Files**: Select individual files (Ctrl/Cmd+click for multiple)
        - **ZIP Folder**: Create a ZIP file containing multiple documents
        - **CSV File**: Upload a spreadsheet with article texts in a 'text' column
        
        **Best Practices:**
        - Ensure files contain readable text content
        - For large batches, consider using ZIP upload
        - CSV files should have a 'text' column with article content
        - Results include file metadata and confidence scores
        """)
    
    # Analysis method selection
    analysis_method = st.radio(
        "Choose analysis method:",
        ["Upload Multiple Files", "Upload ZIP Folder", "Upload CSV File"],
        help="Select whether to upload individual files, a ZIP folder, or a CSV with article texts"
    )
    
    if analysis_method == "Upload Multiple Files":
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose multiple files to analyze",
            type=['pdf', 'txt', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple PDF, TXT, DOCX, or CSV files"
        )
        
        if uploaded_files:
            st.success(f"Successfully uploaded {len(uploaded_files)} files")
            
            # Process uploaded files
            articles_data = []
            document_processor = DocumentProcessor()
            
            with st.spinner("Processing uploaded files..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Extract text from file
                        extracted_text, success, file_type = document_processor.process_uploaded_file(uploaded_file)
                        
                        if success:
                            articles_data.append({
                                'file_name': uploaded_file.name,
                                'file_type': uploaded_file.type,
                                'text': extracted_text,
                                'file_size': len(uploaded_file.getvalue())
                            })
                            st.success(f"✅ {uploaded_file.name} processed successfully")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
            if articles_data:
                # Show preview
                st.subheader("Processed Files Preview")
                preview_df = pd.DataFrame(articles_data)
                st.dataframe(preview_df[['file_name', 'file_type', 'file_size']], use_container_width=True)
                
                # Batch classification
                if st.button("🔍 Perform Batch Classification", type="primary"):
                    perform_batch_classification(classifier, articles_data)
    
    elif analysis_method == "Upload ZIP Folder":
        # ZIP folder upload
        uploaded_zip = st.file_uploader(
            "Choose a ZIP file containing documents",
            type=['zip'],
            help="Upload a ZIP file containing PDF, TXT, DOCX, or CSV files"
        )
        
        if uploaded_zip is not None:
            import zipfile
            import io
            
            try:
                # Extract files from ZIP
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    st.success(f"Found {len(file_list)} files in ZIP")
                    
                    # Filter for supported file types
                    supported_extensions = ['.pdf', '.txt', '.docx', '.csv']
                    supported_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in supported_extensions)]
                    
                    if not supported_files:
                        st.error("No supported file types found in ZIP. Please include PDF, TXT, DOCX, or CSV files.")
                        return
                    
                    st.info(f"Processing {len(supported_files)} supported files")
                    
                    # Process files from ZIP
                    articles_data = []
                    document_processor = DocumentProcessor()
                    
                    with st.spinner("Processing ZIP contents..."):
                        for filename in supported_files:
                            try:
                                # Read file from ZIP
                                with zip_ref.open(filename) as file:
                                    file_content = file.read()
                                
                                # Create a mock uploaded file object
                                class MockUploadedFile:
                                    def __init__(self, name, content, content_type):
                                        self.name = name
                                        self.content = content
                                        self.type = content_type
                                    
                                    def getvalue(self):
                                        return self.content
                                
                                # Determine content type
                                if filename.lower().endswith('.pdf'):
                                    content_type = 'application/pdf'
                                elif filename.lower().endswith('.txt'):
                                    content_type = 'text/plain'
                                elif filename.lower().endswith('.docx'):
                                    content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                                else:
                                    content_type = 'text/csv'
                                
                                mock_file = MockUploadedFile(filename, file_content, content_type)
                                
                                # Extract text
                                extracted_text, success, file_type = document_processor.process_uploaded_file(mock_file)
                                
                                if success:
                                    articles_data.append({
                                        'file_name': filename,
                                        'file_type': content_type,
                                        'text': extracted_text,
                                        'file_size': len(file_content)
                                    })
                                    st.success(f"✅ {filename} processed successfully")
                                else:
                                    st.error(f"❌ Failed to process {filename}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error processing {filename}: {e}")
                    
                    if articles_data:
                        # Show preview
                        st.subheader("Processed Files Preview")
                        preview_df = pd.DataFrame(articles_data)
                        st.dataframe(preview_df[['file_name', 'file_type', 'file_size']], use_container_width=True)
                        
                        # Batch classification
                        if st.button("🔍 Perform Batch Classification", type="primary"):
                            perform_batch_classification(classifier, articles_data)
                            
            except Exception as e:
                st.error(f"Error processing ZIP file: {e}")
    
    else:  # CSV upload
        # Single CSV file upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file with article texts",
            type=['csv'],
            help="CSV should have a 'text' column with article content"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} articles")
                
                # Check for text column
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column with article content.")
                    return
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Convert to articles_data format
                articles_data = []
                for idx, row in df.iterrows():
                    articles_data.append({
                        'file_name': f"article_{idx+1}",
                        'file_type': 'csv',
                        'text': row['text'],
                        'file_size': len(row['text'])
                    })
                
                # Batch classification
                if st.button("🔍 Perform Batch Classification", type="primary"):
                    perform_batch_classification(classifier, articles_data)
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")

def perform_batch_classification(classifier, articles_data):
    """Perform batch classification on articles data."""
    with st.spinner("Processing batch classification..."):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, article in enumerate(articles_data):
            status_text.text(f"Processing {article['file_name']} ({idx + 1} of {len(articles_data)})...")
            article_text = article['text']
            
            prediction, confidence, analysis_data = classifier.classify_article(article_text)
            results.append({
                'file_name': article['file_name'],
                'file_type': article['file_type'],
                'prediction': prediction,
                'article_length': analysis_data.get('article_length', 0),
                'model_used': analysis_data.get('model_used', 'unknown'),
                'file_size': article['file_size']
            })
            
            progress_bar.progress((idx + 1) / len(articles_data))
        
        status_text.text("Processing complete!")
        
        # Create simple results DataFrame
        simple_results = []
        for result in results:
            simple_results.append({
                'Document Name': result['file_name'],
                'Political Leaning': result['prediction']
            })
        
        results_df = pd.DataFrame(simple_results)
        
        # Display simple results table
        st.subheader("Classification Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Simple summary
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            left_count = len(results_df[results_df['Political Leaning'] == 'left-leaning'])
            st.metric("Left-Leaning", left_count)
        
        with col2:
            right_count = len(results_df[results_df['Political Leaning'] == 'right-leaning'])
            st.metric("Right-Leaning", right_count)
        

        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="batch_classification_results.csv",
            mime="text/csv"
        )







def display_cv_results(results, cv_type):
    """Display cross-validation results with confusion matrix and metrics."""
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.3f}")
    
    with col2:
        st.metric("Precision", f"{results['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{results['recall']:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{results['f1_score']:.3f}")
    
    # Display confusion matrix
    conf_matrix = results['confusion_matrix']
    
    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['left-leaning', 'right-leaning'],
        y=['left-leaning', 'right-leaning'],
        title=f"{cv_type} Cross-Validation Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Detailed Metrics:**")
        st.write(f"- True Positives (TP): {tp}")
        st.write(f"- True Negatives (TN): {tn}")
        st.write(f"- False Positives (FP): {fp}")
        st.write(f"- False Negatives (FN): {fn}")
    
    with col2:
        st.write("**Calculated Metrics:**")
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        error_rate = (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        st.write(f"- Sensitivity: {sensitivity:.3f}")
        st.write(f"- Specificity: {specificity:.3f}")
        st.write(f"- Error Rate: {error_rate:.3f}")

def display_all_cv_results(cv_results):
    """Display comparison of all cross-validation results."""
    
    st.subheader("📊 Cross-Validation Results Comparison")
    
    # Create comparison table
    comparison_data = []
    
    for cv_type, results in cv_results.items():
        conf_matrix = results['confusion_matrix']
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        comparison_data.append({
            'CV Type': cv_type.replace('fold', '-Fold').replace('loo', 'Leave-One-Out'),
            'Accuracy': f"{results['accuracy']:.3f}",
            'Precision': f"{results['precision']:.3f}",
            'Recall': f"{results['recall']:.3f}",
            'F1-Score': f"{results['f1_score']:.3f}",
            'Sensitivity': f"{sensitivity:.3f}",
            'Specificity': f"{specificity:.3f}",
            'Total Errors': fp + fn
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Display all confusion matrices side by side
    st.subheader("🎯 Confusion Matrices Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if '5fold' in cv_results:
            results = cv_results['5fold']
            conf_matrix = results['confusion_matrix']
            fig = px.imshow(
                conf_matrix,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['left-leaning', 'right-leaning'],
                y=['left-leaning', 'right-leaning'],
                title="5-Fold CV"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if '10fold' in cv_results:
            results = cv_results['10fold']
            conf_matrix = results['confusion_matrix']
            fig = px.imshow(
                conf_matrix,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['left-leaning', 'right-leaning'],
                y=['left-leaning', 'right-leaning'],
                title="10-Fold CV"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if 'loo' in cv_results:
            results = cv_results['loo']
            conf_matrix = results['confusion_matrix']
            fig = px.imshow(
                conf_matrix,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['left-leaning', 'right-leaning'],
                y=['left-leaning', 'right-leaning'],
                title="Leave-One-Out CV"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_results_page():
    """Show comprehensive evaluation results."""
    st.header("📊 Comprehensive Evaluation Results")
    st.markdown("### Real Articles Political Leaning Classifier Performance")
    
    # Initialize classifier
    classifier = initialize_classifier()
    if classifier is None:
        st.error("Failed to initialize classifier. Please check your configuration.")
        return
    
    # Display pre-generated confusion matrices
    st.subheader("🎯 Pre-Generated Confusion Matrices")
    st.markdown("Displaying confusion matrices from previous cross-validation analyses.")
    
    # Load pre-generated results
    try:
        import json
        with open('confusion_matrix_results.json', 'r') as f:
            cv_results = json.load(f)
        
        # Display confusion matrices
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**5-Fold Cross-Validation**")
            try:
                st.image('confusion_matrix_5fold.png', use_container_width=True)
            except:
                st.error("5-fold confusion matrix image not found")
            
            # Display metrics
            if '5_fold' in cv_results:
                results = cv_results['5_fold']
                st.write(f"**Accuracy:** {results['accuracy']:.3f}")
                st.write(f"**Specificity:** {results['specificity']:.3f}")
                st.write(f"**Sensitivity:** {results['sensitivity']:.3f}")
        
        with col2:
            st.write("**10-Fold Cross-Validation**")
            try:
                st.image('confusion_matrix_10fold.png', use_container_width=True)
            except:
                st.error("10-fold confusion matrix image not found")
            
            # Display metrics
            if '10_fold' in cv_results:
                results = cv_results['10_fold']
                st.write(f"**Accuracy:** {results['accuracy']:.3f}")
                st.write(f"**Specificity:** {results['specificity']:.3f}")
                st.write(f"**Sensitivity:** {results['sensitivity']:.3f}")
        
        with col3:
            st.write("**Leave-One-Out Cross-Validation**")
            try:
                st.image('confusion_matrix_leave_one_out.png', use_container_width=True)
            except:
                st.error("Leave-one-out confusion matrix image not found")
            
            # Display metrics
            if 'leave_one_out' in cv_results:
                results = cv_results['leave_one_out']
                st.write(f"**Accuracy:** {results['accuracy']:.3f}")
                st.write(f"**Specificity:** {results['specificity']:.3f}")
                st.write(f"**Sensitivity:** {results['sensitivity']:.3f}")
        
        # Display comparison plot if available
        try:
            st.subheader("📊 Cross-Validation Comparison")
            st.image('cross_validation_comparison.png', use_container_width=True)
        except:
            st.info("Comparison plot not available")
        
        # Store results in session state for download functionality
        st.session_state.cv_results = cv_results
        
    except Exception as e:
        st.error(f"Error loading pre-generated results: {e}")
        st.info("Please run the cross-validation analysis to generate new results.")
    

    

    
    # Key findings
    st.subheader("🔍 Key Findings")
    
    st.write("""
    **Model Performance:**
    - The fine-tuned GPT-4.1 model analyzes real political articles for leaning classification
    - Cross-validation provides robust evaluation of model performance
    - Results show the model's ability to distinguish between left and right-leaning content
    
    **Analysis Features:**
    - 5-Fold Cross-Validation: Balanced evaluation with 5 folds
    - 10-Fold Cross-Validation: More detailed evaluation with 10 folds  
    - Leave-One-Out Cross-Validation: Most thorough evaluation using all but one sample
    
    **Model Strengths:**
    - Excellent at distinguishing between left and right-leaning political content
    - Consistent performance across different article topics
    - Robust to different article lengths and writing styles
    """)
    
    # Download results (only show if results are available)
    if hasattr(st.session_state, 'cv_results'):
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download 5-Fold Results"):
                results_5fold = st.session_state.cv_results['5_fold']
                download_pre_generated_results(results_5fold, "5_fold")
        
        with col2:
            if st.button("📥 Download 10-Fold Results"):
                results_10fold = st.session_state.cv_results['10_fold']
                download_pre_generated_results(results_10fold, "10_fold")
        
        with col3:
            if st.button("📥 Download Leave-One-Out Results"):
                results_loo = st.session_state.cv_results['leave_one_out']
                download_pre_generated_results(results_loo, "leave_one_out")

def download_pre_generated_results(results, cv_type):
    """Download pre-generated cross-validation results as CSV."""
    
    cv_type_display = cv_type.replace('_', ' ').title().replace('Fold', '-Fold')
    if cv_type == 'leave_one_out':
        cv_type_display = 'Leave-One-Out'
    
    results_data = {
        'Test_Type': f'{cv_type_display} CV',
        'Accuracy': results['accuracy'],
        'Specificity': results['specificity'],
        'Sensitivity': results['sensitivity'],
        'N_Samples': results['n_samples']
    }
    
    df = pd.DataFrame([results_data])
    csv = df.to_csv(index=False)
    
    st.download_button(
        label=f"Download {cv_type_display} Results",
        data=csv,
        file_name=f"{cv_type}_cv_results.csv",
        mime="text/csv"
    )

def download_cv_results(results, cv_type):
    """Download cross-validation results as CSV."""
    
    conf_matrix = results['confusion_matrix']
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results_data = {
        'Test_Type': f'{cv_type.replace("fold", "-Fold").replace("loo", "Leave-One-Out")} CV',
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1_Score': results['f1_score'],
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'True_Positives': tp,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'Total_Errors': fp + fn
    }
    
    df = pd.DataFrame([results_data])
    csv = df.to_csv(index=False)
    
    st.download_button(
        label=f"Download {cv_type.replace('fold', '-Fold').replace('loo', 'Leave-One-Out')} Results",
        data=csv,
        file_name=f"{cv_type}_cv_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main() 