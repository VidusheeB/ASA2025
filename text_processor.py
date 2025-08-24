import re
import json
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PoliticalTextProcessor:
    """
    Process article text to extract TF-IDF scores for political terms.
    """
    
    def __init__(self):
        """Initialize with political terms vocabulary."""
        self.political_terms = [
            "Scrutiny", "hostile attitudes", "national security", "political activism",
            "privilege", "authoritarianism", "suppression", "disinformation",
            "unlawful", "vetting", "inhumane conditions", "rights violations",
            "deportation", "denaturalization", "fraud", "abuse", "exploitation",
            "fear", "misinformation", "cruelty", "unconstitutional", "terror network", "chaos"
        ]
        
        # Create TF-IDF vectorizer with political terms
        self.vectorizer = TfidfVectorizer(
            vocabulary={term.lower(): i for i, term in enumerate(self.political_terms)},
            lowercase=True,
            ngram_range=(1, 2),  # Allow unigrams and bigrams
            min_df=1,
            max_df=1.0
        )
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TF-IDF analysis.
        
        Args:
            text: Raw article text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_tfidf_scores(self, article_text: str) -> Dict[str, float]:
        """
        Extract TF-IDF scores for political terms from article text.
        
        Args:
            article_text: The article text to analyze
            
        Returns:
            Dictionary of political terms and their TF-IDF scores
        """
        # Preprocess the text
        processed_text = self.preprocess_text(article_text)
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform([processed_text])
        
        # Get feature names (terms)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Extract scores for political terms
        scores = {}
        for i, term in enumerate(self.political_terms):
            term_lower = term.lower()
            if term_lower in feature_names:
                # Find the index of the term in feature names
                try:
                    idx = list(feature_names).index(term_lower)
                    score = tfidf_matrix[0, idx]
                    scores[term] = float(score)
                except ValueError:
                    scores[term] = 0.0
            else:
                scores[term] = 0.0
        
        return scores
    
    def analyze_article(self, article_text: str) -> Dict:
        """
        Analyze an article and return TF-IDF scores with analysis.
        
        Args:
            article_text: The article text to analyze
            
        Returns:
            Dictionary with TF-IDF scores and analysis
        """
        # Extract TF-IDF scores
        tfidf_scores = self.extract_tfidf_scores(article_text)
        
        # Find terms with highest scores
        significant_terms = {term: score for term, score in tfidf_scores.items() if score > 0}
        sorted_terms = sorted(significant_terms.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'tfidf_scores': tfidf_scores,
            'significant_terms': sorted_terms[:10],  # Top 10 terms
            'total_terms_found': len(significant_terms),
            'max_score': max(tfidf_scores.values()) if tfidf_scores.values() else 0
        } 