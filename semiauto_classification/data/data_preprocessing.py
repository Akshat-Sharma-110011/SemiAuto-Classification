"""
Data Preprocessing Module for SemiAuto-Classification

This module handles the preprocessing of data for the classification model, including:
- Handling missing values
- Handling duplicate values
- Handling outliers
- Handling skewed data
- Scaling numerical features
- Encoding categorical features
- Text preprocessing (cleaning, tokenization, vectorization)

The preprocessing steps are configured based on information in the feature_store.yaml file,
and the preprocessing pipeline is saved for later use.
"""

"""
CRITICAL FIX: Data Preprocessing Module with Proper Text Processing

Key changes made:
1. Fixed TextPreprocessor to ensure text columns are actually transformed
2. Enhanced the transform method to apply text preprocessing first
3. Added proper column handling and feature generation
4. Improved logging and error handling
5. Ensured text columns are removed after processing
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import cloudpickle
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
import scipy.stats as stats
from pydantic import BaseModel
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import gensim
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up the logger
from semiauto_classification.logger import section, configure_logger
from semiauto_classification.custom_transformers import (IDColumnDropper, MissingValueHandler,
                                                         CategoricalEncoder, NumericalScaler, SkewedDataHandler,
                                                         OutlierHandler)

# Configure logger
configure_logger()
logger = logging.getLogger("Data Preprocessing")


def get_dataset_name():
    """Lazily load dataset name when needed"""
    try:
        with open('intel.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config['dataset_name']
    except FileNotFoundError:
        return "default_dataset"


dataset_name = get_dataset_name()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    FIXED: Enhanced TextPreprocessor that ensures text columns are properly transformed
    """

    def __init__(
            self,
            columns: List[str] = None,
            lowercase: bool = True,
            remove_html: bool = True,
            remove_urls: bool = True,
            handle_emojis: str = 'remove',
            remove_punctuation: bool = True,
            handle_chat_words: bool = False,
            spelling_correction: bool = False,
            remove_stopwords: bool = True,
            stemming_lemmatization: str = 'none',
            pos_tagging: bool = False,
            tokenization_method: str = 'tfidf',
            ngram_range: Tuple[int, int] = (1, 2),
            max_features: int = 1000,
            vector_size: int = 100
    ):
        self.columns = columns or []
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.handle_emojis = handle_emojis
        self.remove_punctuation = remove_punctuation
        self.handle_chat_words = handle_chat_words
        self.spelling_correction = spelling_correction
        self.remove_stopwords = remove_stopwords
        self.stemming_lemmatization = stemming_lemmatization
        self.pos_tagging = pos_tagging
        self.tokenization_method = tokenization_method
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vector_size = vector_size

        # Initialize processors
        try:
            self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        except:
            self.stop_words = set()
            logger.warning("Could not load stopwords, continuing without them")

        self.stemmer = PorterStemmer() if stemming_lemmatization == 'stemming' else None
        self.lemmatizer = WordNetLemmatizer() if stemming_lemmatization == 'lemmatization' else None

        # Vectorizers and models - store per column
        self.vectorizers = {}
        self.word2vec_models = {}
        self.fitted_columns = []
        self.feature_names = {}

        # Chat words dictionary
        self.chat_words_dict = {
            "u": "you", "r": "are", "ur": "your", "n": "and", "2": "to", "4": "for",
            "b4": "before", "gr8": "great", "m8": "mate", "w8": "wait", "h8": "hate",
            "luv": "love", "plz": "please", "thx": "thanks", "omg": "oh my god",
            "lol": "laugh out loud", "brb": "be right back", "ttyl": "talk to you later",
            "btw": "by the way", "imo": "in my opinion", "imho": "in my humble opinion",
            "fyi": "for your information", "asap": "as soon as possible", "w/": "with",
            "w/o": "without", "ppl": "people", "bf": "boyfriend", "gf": "girlfriend",
            "gonna": "going to", "wanna": "want to", "gotta": "got to", "lemme": "let me",
            "gimme": "give me", "kinda": "kind of", "sorta": "sort of", "outta": "out of",
            "shoulda": "should have", "coulda": "could have", "woulda": "would have",
            "ain't": "is not", "can't": "cannot", "won't": "will not", "don't": "do not",
            "didn't": "did not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not"
        }

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags"""
        if not self.remove_html:
            return text
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub('', text)

    def _clean_urls(self, text: str) -> str:
        """Remove URLs"""
        if not self.remove_urls:
            return text
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = url_pattern.sub('', text)
        www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return www_pattern.sub('', text)

    def _handle_emojis(self, text: str) -> str:
        """Handle emojis based on configuration"""
        if self.handle_emojis == 'keep':
            return text

        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )

        if self.handle_emojis == 'remove':
            return emoji_pattern.sub('', text)
        elif self.handle_emojis == 'replace':
            return emoji_pattern.sub('[EMOJI]', text)
        return text

    def _expand_chat_words(self, text: str) -> str:
        """Expand chat words and contractions"""
        if not self.handle_chat_words:
            return text

        words = text.split()
        expanded_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in self.chat_words_dict:
                expanded_words.append(self.chat_words_dict[lower_word])
            else:
                expanded_words.append(word)
        return ' '.join(expanded_words)

    def _correct_spelling(self, text: str) -> str:
        """Basic spelling correction (simplified implementation)"""
        if not self.spelling_correction:
            return text
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return str(blob.correct())
        except ImportError:
            logger.warning("TextBlob not available for spelling correction")
            return text

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation"""
        if not self.remove_punctuation:
            return text
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def _remove_stopwords_func(self, text: str) -> str:
        """Remove stopwords"""
        if not self.remove_stopwords or not self.stop_words:
            return text

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def _get_wordnet_pos(self, word):
        """Map POS tag to first character used by WordNetLemmatizer"""
        try:
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        except:
            return wordnet.NOUN

    def _stem_or_lemmatize(self, text: str) -> str:
        """Apply stemming or lemmatization"""
        if self.stemming_lemmatization == 'none':
            return text

        words = text.split()

        if self.stemming_lemmatization == 'stemming' and self.stemmer:
            return ' '.join([self.stemmer.stem(word) for word in words])
        elif self.stemming_lemmatization == 'lemmatization' and self.lemmatizer:
            return ' '.join([self.lemmatizer.lemmatize(word, self._get_wordnet_pos(word)) for word in words])

        return text

    def _extract_pos_tags(self, text: str) -> str:
        """Extract POS tags and append to text"""
        if not self.pos_tagging:
            return text

        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            pos_features = [f"{word}_{tag}" for word, tag in pos_tags]
            return ' '.join(pos_features)
        except:
            return text

    def _preprocess_text(self, text: str) -> str:
        """Apply all text preprocessing steps"""
        if pd.isna(text) or text == '' or text is None:
            return ''

        text = str(text)

        # Apply preprocessing steps in order
        text = self._clean_html(text)
        text = self._clean_urls(text)
        text = self._handle_emojis(text)
        text = self._expand_chat_words(text)
        text = self._correct_spelling(text)

        if self.lowercase:
            text = text.lower()

        text = self._remove_punctuation(text)
        text = self._remove_stopwords_func(text)
        text = self._stem_or_lemmatize(text)
        text = self._extract_pos_tags(text)

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the text preprocessor"""
        logger.info(f"Fitting TextPreprocessor on columns: {self.columns}")

        # Store fitted columns
        self.fitted_columns = [col for col in self.columns if col in X.columns]

        if not self.fitted_columns:
            logger.warning("No valid text columns found for preprocessing")
            return self

        logger.info(f"Processing {len(self.fitted_columns)} text columns: {self.fitted_columns}")

        # Process each column separately
        for col in self.fitted_columns:
            logger.info(f"Fitting text preprocessor for column: {col}")

            # Preprocess all text for this column
            col_texts = X[col].apply(self._preprocess_text).tolist()

            # Filter out empty texts
            processed_texts = [text for text in col_texts if text and text.strip()]

            if not processed_texts:
                logger.warning(f"No valid text content found after preprocessing in column {col}")
                continue

            logger.info(f"Processed {len(processed_texts)} valid texts in column {col}")

            # Set up vectorizer based on tokenization method
            if self.tokenization_method in ['bow', 'tfidf', 'ngrams']:
                try:
                    if self.tokenization_method == 'bow':
                        vectorizer = CountVectorizer(
                            max_features=self.max_features,
                            ngram_range=self.ngram_range if self.tokenization_method == 'ngrams' else (1, 1)
                        )
                    else:  # tfidf or ngrams
                        vectorizer = TfidfVectorizer(
                            max_features=self.max_features,
                            ngram_range=self.ngram_range if self.tokenization_method == 'ngrams' else (1, 1)
                        )

                    vectorizer.fit(processed_texts)
                    self.vectorizers[col] = vectorizer
                    self.feature_names[col] = vectorizer.get_feature_names_out()
                    logger.info(
                        f"Fitted {self.tokenization_method} vectorizer for {col} with {len(self.feature_names[col])} features")

                except Exception as e:
                    logger.error(f"Error fitting vectorizer for column {col}: {str(e)}")

            elif self.tokenization_method in ['word2vec', 'glove']:
                try:
                    # Tokenize texts for Word2Vec
                    tokenized_texts = [text.split() for text in processed_texts if text.strip()]
                    if tokenized_texts and len(tokenized_texts) > 0:
                        model = Word2Vec(
                            sentences=tokenized_texts,
                            vector_size=self.vector_size,
                            window=5,
                            min_count=1,
                            workers=4,
                            seed=42
                        )
                        self.word2vec_models[col] = model
                        logger.info(f"Fitted Word2Vec model for {col} with {self.vector_size} dimensions")
                    else:
                        logger.warning(f"No tokenized texts available for Word2Vec in column {col}")

                except Exception as e:
                    logger.error(f"Error fitting Word2Vec model for column {col}: {str(e)}")

        logger.info(f"TextPreprocessor fitting completed with method: {self.tokenization_method}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the text columns - CRITICAL FIX"""
        if not self.fitted_columns:
            logger.warning("No fitted columns for text transformation")
            return X

        X_transformed = X.copy()
        logger.info(f"Starting text transformation for columns: {self.fitted_columns}")

        for col in self.fitted_columns:
            if col not in X_transformed.columns:
                logger.warning(f"Column {col} not found in transform data")
                continue

            logger.info(f"Transforming text column: {col} using method: {self.tokenization_method}")

            # Preprocess text
            processed_texts = X_transformed[col].apply(self._preprocess_text)
            logger.info(f"Preprocessed {len(processed_texts)} texts for column {col}")

            if self.tokenization_method == 'none':
                # Just keep the preprocessed text
                X_transformed[f"{col}_processed"] = processed_texts
                logger.info(f"Applied basic text cleaning to {col}")

            elif self.tokenization_method in ['bow', 'tfidf', 'ngrams'] and col in self.vectorizers:
                try:
                    # Transform all texts at once for efficiency
                    vectorizer = self.vectorizers[col]
                    vectors = vectorizer.transform(processed_texts).toarray()

                    # Create feature columns with safe names
                    feature_names = self.feature_names[col]
                    for i, feature_name in enumerate(feature_names):
                        # Create safe column names
                        safe_name = str(feature_name).replace(' ', '_').replace('-', '_').replace('.', '_')
                        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', safe_name)
                        column_name = f"{col}_{safe_name}"
                        X_transformed[column_name] = vectors[:, i]

                    logger.info(f"Created {len(feature_names)} vectorized features from {col}")

                except Exception as e:
                    logger.error(f"Error in vectorization for {col}: {str(e)}")
                    # Fallback to processed text
                    X_transformed[f"{col}_processed"] = processed_texts

            elif self.tokenization_method in ['word2vec', 'glove'] and col in self.word2vec_models:
                try:
                    model = self.word2vec_models[col]
                    embeddings = []

                    for text in processed_texts:
                        if not text or not text.strip():
                            embeddings.append(np.zeros(self.vector_size))
                            continue

                        words = text.split()
                        vectors = []
                        for word in words:
                            try:
                                vectors.append(model.wv[word])
                            except KeyError:
                                pass  # Word not in vocabulary

                        if vectors:
                            embedding = np.mean(vectors, axis=0)
                        else:
                            embedding = np.zeros(self.vector_size)
                        embeddings.append(embedding)

                    embeddings = np.array(embeddings)

                    # Create embedding feature columns
                    for i in range(embeddings.shape[1]):
                        X_transformed[f"{col}_embed_{i}"] = embeddings[:, i]

                    logger.info(f"Created {embeddings.shape[1]} embedding features from {col}")

                except Exception as e:
                    logger.error(f"Error in word2vec transformation for {col}: {str(e)}")
                    X_transformed[f"{col}_processed"] = processed_texts

            else:
                # If no specific method available, keep processed text
                X_transformed[f"{col}_processed"] = processed_texts
                logger.info(f"Applied basic preprocessing to {col}")

            # CRITICAL: Remove original text column
            X_transformed.drop(columns=[col], inplace=True)
            logger.info(f"Removed original text column: {col}")

        new_features = [c for c in X_transformed.columns if c not in X.columns]
        logger.info(f"Text preprocessing completed. Generated {len(new_features)} new features")
        return X_transformed

# Fix for the main preprocessing pipeline
def fix_preprocessing_pipeline_text_handling():
    """
    Fix to ensure text preprocessing is properly applied in the main pipeline
    """

    # This is the key fix for the PreprocessingPipeline class
    def enhanced_transform(self, X: pd.DataFrame, handle_duplicates: bool = True) -> pd.DataFrame:
        """Enhanced transform method that ensures text preprocessing is applied"""
        logger.info("Transforming data with preprocessing pipeline")
        transformed_data = X.copy()

        # Apply ID column dropper first
        if self.id_dropper:
            transformed_data = self.id_dropper.transform(transformed_data)

        # Handle duplicates if requested
        if handle_duplicates:
            transformed_data = self.remove_duplicates(transformed_data)

        # Extract target data BEFORE text processing to avoid issues
        if self.target_column and self.target_column in transformed_data.columns:
            target_data = transformed_data.pop(self.target_column)
        else:
            target_data = None

        # CRITICAL: Apply text preprocessing FIRST before other transformations
        # This ensures text columns are converted to numerical features early
        if self.text_preprocessor:
            logger.info("Applying text preprocessing...")
            original_cols = set(transformed_data.columns)
            transformed_data = self.text_preprocessor.transform(transformed_data)
            new_cols = set(transformed_data.columns) - original_cols
            logger.info(f"Text preprocessing created {len(new_cols)} new features")

            # Update column lists for subsequent transformations
            # Remove processed text columns from numerical/categorical lists if they were there
            if hasattr(self, 'missing_handler') and self.missing_handler:
                text_cols = self.text_preprocessor.fitted_columns
                self.missing_handler.columns = [c for c in self.missing_handler.columns if c not in text_cols]

            if hasattr(self, 'numerical_scaler') and self.numerical_scaler:
                text_cols = self.text_preprocessor.fitted_columns
                self.numerical_scaler.columns = [c for c in self.numerical_scaler.columns if c not in text_cols]
                # Add new numerical features from text processing
                new_numerical_cols = [c for c in new_cols if
                                      c in transformed_data.select_dtypes(include=[np.number]).columns]
                self.numerical_scaler.columns.extend(new_numerical_cols)

        # Transform with missing value handler if defined
        if self.missing_handler:
            transformed_data = self.missing_handler.transform(transformed_data)

        # Transform with outlier handler if defined
        if self.outlier_handler:
            transformed_data = self.outlier_handler.transform(transformed_data)

        # Transform with skewed data handler if defined
        if self.skewed_handler:
            transformed_data = self.skewed_handler.transform(transformed_data)

        # Transform with numerical scaler if defined
        if self.numerical_scaler:
            transformed_data = self.numerical_scaler.transform(transformed_data)

        # Transform with categorical encoder if defined
        if self.categorical_encoder:
            transformed_data = self.categorical_encoder.transform(transformed_data)

        # Handle target column encoding and restoration
        if target_data is not None:
            if self.target_encoder is not None:
                if target_data.isnull().any():
                    logger.error("Target column contains missing values after preprocessing")
                    raise ValueError("Target column contains missing values")
                encoded_target = self.target_encoder.transform(target_data)
                transformed_data[self.target_column] = encoded_target
                logger.info(f"Encoded target column '{self.target_column}' using LabelEncoder")
            else:
                transformed_data[self.target_column] = pd.to_numeric(target_data, errors='coerce')
                if transformed_data[self.target_column].isnull().any():
                    logger.error("Target contains non-numeric values that couldn't be coerced")
                    raise ValueError("Invalid values in target column")
                logger.info(f"Converted target column '{self.target_column}' to numeric")

        # Ensure target column is the last column
        if self.target_column and self.target_column in transformed_data.columns:
            cols = [col for col in transformed_data.columns if col != self.target_column] + [self.target_column]
            transformed_data = transformed_data[cols]

        return transformed_data

    return enhanced_transform


# Additional helper functions to ensure text processing works correctly
def validate_text_preprocessing_config(text_config: dict, text_columns: list) -> dict:
    """
    Validate and ensure text preprocessing configuration is properly set up
    """
    if not text_columns:
        logger.warning("No text columns provided for preprocessing")
        return text_config

    # Ensure essential settings are enabled for actual text processing
    validated_config = text_config.copy()

    # If tokenization method is 'none', still apply basic text cleaning
    if validated_config.get('tokenization_method') == 'none':
        logger.info("Tokenization method is 'none' - will apply basic text cleaning only")

    # Ensure at least some preprocessing is happening
    if not any([
        validated_config.get('lowercase', False),
        validated_config.get('remove_punctuation', False),
        validated_config.get('remove_stopwords', False),
        validated_config.get('tokenization_method') != 'none'
    ]):
        logger.warning("No text preprocessing options enabled - enabling basic cleaning")
        validated_config.update({
            'lowercase': True,
            'remove_punctuation': True,
            'tokenization_method': 'tfidf'
        })

    logger.info(f"Text preprocessing config validated for {len(text_columns)} columns")
    return validated_config


def debug_text_preprocessing(df: pd.DataFrame, text_columns: list):
    """
    Debug function to check text preprocessing steps
    """
    logger.info("=== TEXT PREPROCESSING DEBUG ===")

    for col in text_columns:
        if col in df.columns:
            sample_texts = df[col].dropna().head(3).tolist()
            logger.info(f"Column '{col}' sample texts:")
            for i, text in enumerate(sample_texts):
                logger.info(f"  {i + 1}: {str(text)[:100]}...")
        else:
            logger.warning(f"Text column '{col}' not found in dataframe")

    logger.info("=== END TEXT PREPROCESSING DEBUG ===")


class PreprocessingParameters(BaseModel):
    """Pydantic model for preprocessing parameters"""
    # Missing Values Handling
    missing_values_method: str = 'mean'
    missing_values_columns: List[str] = []

    # Duplicates Handling
    handle_duplicates: bool = True

    # Outlier Handling
    outliers_method: Optional[str] = None
    outliers_columns: List[str] = []

    # Skewness Handling
    skewness_method: Optional[str] = None
    skewness_columns: List[str] = []

    # Numerical Scaling
    scaling_method: Optional[str] = None
    scaling_columns: List[str] = []

    # Categorical Encoding
    categorical_encoding_method: Optional[str] = None
    categorical_columns: List[str] = []
    drop_first: bool = True

    # Text Preprocessing
    text_preprocessing_enabled: bool = False
    text_columns: List[str] = []
    text_preprocessing_config: Optional[Dict[str, Any]] = None


class PreprocessingPipeline:
    """ENHANCED: API-friendly preprocessing pipeline with FIXED text processing"""

    def __init__(self, config: Dict[str, Any], params: PreprocessingParameters = None):
        """Initialize with both original config and API parameters"""
        self.config = get_intel_config()
        self.params = params or PreprocessingParameters()
        self.dataset_name = config.get('dataset_name')
        self.target_column = config.get('target_col')
        self.feature_store = config.get('feature_store', {})

        # Initialize handlers
        self.missing_handler = None
        self.outlier_handler = None
        self.skewed_handler = None
        self.numerical_scaler = None
        self.categorical_encoder = None
        self.text_preprocessor = None
        self.id_dropper = IDColumnDropper(
            id_cols=self.feature_store.get('id_cols', [])
        )
        self.target_encoder = None

        logger.info(f"Initialized API PreprocessingPipeline for dataset: {self.dataset_name}")

    def configure_pipeline(self):
        """Configure pipeline based on received parameters"""
        # Filter out target column from all preprocessing columns
        target_col = self.target_column

        # Missing values handling
        missing_cols = [col for col in self.params.missing_values_columns if col != target_col]
        if missing_cols:
            self.missing_handler = MissingValueHandler(
                method=self.params.missing_values_method,
                columns=missing_cols
            )

        # Initialize ID dropper with config from feature store
        self.id_dropper = IDColumnDropper(
            id_cols=self.feature_store.get('id_cols', [])
        )

        # Outlier handling
        outlier_cols = [col for col in self.params.outliers_columns if col != target_col]
        if self.params.outliers_method and outlier_cols:
            self.outlier_handler = OutlierHandler(
                method=self.params.outliers_method,
                columns=outlier_cols
            )

        # Skewed data handling
        skewed_cols = [col for col in self.params.skewness_columns if col != target_col]
        if self.params.skewness_method and skewed_cols:
            self.skewed_handler = SkewedDataHandler(
                method=self.params.skewness_method,
                columns=skewed_cols
            )

        # Numerical scaling
        scaling_cols = [col for col in self.params.scaling_columns if col != target_col]
        if self.params.scaling_method and scaling_cols:
            self.numerical_scaler = NumericalScaler(
                method=self.params.scaling_method,
                columns=scaling_cols
            )

        # Categorical encoding
        categorical_cols = [col for col in self.params.categorical_columns if col != target_col]
        if self.params.categorical_encoding_method and categorical_cols:
            self.categorical_encoder = CategoricalEncoder(
                method=self.params.categorical_encoding_method,
                columns=categorical_cols,
                drop_first=self.params.drop_first
            )

        # CRITICAL: Text preprocessing setup
        if self.params.text_preprocessing_enabled and self.params.text_columns:
            text_cols = [col for col in self.params.text_columns if col != target_col]
            if text_cols and self.params.text_preprocessing_config:
                logger.info(f"Setting up text preprocessing for columns: {text_cols}")
                logger.info(f"Text config: {self.params.text_preprocessing_config}")

                self.text_preprocessor = TextPreprocessor(
                    columns=text_cols,
                    lowercase=self.params.text_preprocessing_config.get('lowercase', True),
                    remove_html=self.params.text_preprocessing_config.get('remove_html', True),
                    remove_urls=self.params.text_preprocessing_config.get('remove_urls', True),
                    handle_emojis=self.params.text_preprocessing_config.get('handle_emojis', 'remove'),
                    remove_punctuation=self.params.text_preprocessing_config.get('remove_punctuation', True),
                    handle_chat_words=self.params.text_preprocessing_config.get('handle_chat_words', False),
                    spelling_correction=self.params.text_preprocessing_config.get('spelling_correction', False),
                    remove_stopwords=self.params.text_preprocessing_config.get('remove_stopwords', True),
                    stemming_lemmatization=self.params.text_preprocessing_config.get('stemming_lemmatization', 'none'),
                    pos_tagging=self.params.text_preprocessing_config.get('pos_tagging', False),
                    tokenization_method=self.params.text_preprocessing_config.get('tokenization_method', 'tfidf'),
                    ngram_range=tuple(self.params.text_preprocessing_config.get('ngram_range', [1, 2])),
                    max_features=self.params.text_preprocessing_config.get('max_features', 1000),
                    vector_size=self.params.text_preprocessing_config.get('vector_size', 100)
                )
                logger.info(f"Text preprocessor configured for {len(text_cols)} columns")

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the data"""
        rows_before = len(df)
        df_no_duplicates = df.drop_duplicates()
        rows_after = len(df_no_duplicates)
        rows_dropped = rows_before - rows_after

        if rows_dropped > 0:
            logger.info(f"Removed {rows_dropped} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        return df_no_duplicates

    def fit(self, X: pd.DataFrame) -> None:
        """Fit the preprocessing pipeline on the training data"""
        logger.info("Fitting preprocessing pipeline")

        if self.id_dropper:
            self.id_dropper.fit(X)

        # CRITICAL: Fit text preprocessor FIRST to understand text columns
        if self.text_preprocessor:
            logger.info("Fitting text preprocessor...")
            self.text_preprocessor.fit(X)
            logger.info("Text preprocessor fitted successfully")

        # Fit other handlers
        if self.missing_handler:
            self.missing_handler.fit(X)

        if self.outlier_handler:
            self.outlier_handler.fit(X)

        if self.skewed_handler:
            self.skewed_handler.fit(X)

        if self.numerical_scaler:
            self.numerical_scaler.fit(X)

        if self.categorical_encoder:
            self.categorical_encoder.fit(X)

        # Handle target encoding
        if self.target_column and self.target_column in X.columns:
            target_data = X[self.target_column]
            if not pd.api.types.is_numeric_dtype(target_data):
                self.target_encoder = LabelEncoder()
                self.target_encoder.fit(target_data)
                logger.info(f"Fitted LabelEncoder on target column '{self.target_column}'")

    def transform(self, X: pd.DataFrame, handle_duplicates: bool = True) -> pd.DataFrame:
        """CRITICAL FIX: Enhanced transform method that ensures text preprocessing is applied"""
        logger.info("Transforming data with preprocessing pipeline")
        transformed_data = X.copy()

        # Apply ID column dropper first
        if self.id_dropper:
            transformed_data = self.id_dropper.transform(transformed_data)

        # Handle duplicates if requested
        if handle_duplicates:
            transformed_data = self.remove_duplicates(transformed_data)

        # Extract target data BEFORE any transformations
        if self.target_column and self.target_column in transformed_data.columns:
            target_data = transformed_data.pop(self.target_column)
        else:
            target_data = None

        # CRITICAL FIX: Apply text preprocessing FIRST
        if self.text_preprocessor:
            logger.info("=== APPLYING TEXT PREPROCESSING ===")
            original_shape = transformed_data.shape
            original_cols = set(transformed_data.columns)

            transformed_data = self.text_preprocessor.transform(transformed_data)

            new_cols = set(transformed_data.columns) - original_cols
            logger.info(f"Text preprocessing: {original_shape} -> {transformed_data.shape}")
            logger.info(f"Text preprocessing created {len(new_cols)} new features")
            logger.info("=== TEXT PREPROCESSING COMPLETED ===")

        # Apply other transformations
        if self.missing_handler:
            transformed_data = self.missing_handler.transform(transformed_data)

        if self.outlier_handler:
            transformed_data = self.outlier_handler.transform(transformed_data)

        if self.skewed_handler:
            transformed_data = self.skewed_handler.transform(transformed_data)

        if self.numerical_scaler:
            transformed_data = self.numerical_scaler.transform(transformed_data)

        if self.categorical_encoder:
            transformed_data = self.categorical_encoder.transform(transformed_data)

        # Handle target column restoration
        if target_data is not None:
            if self.target_encoder is not None:
                if target_data.isnull().any():
                    logger.error("Target column contains missing values after preprocessing")
                    raise ValueError("Target column contains missing values")
                encoded_target = self.target_encoder.transform(target_data)
                transformed_data[self.target_column] = encoded_target
                logger.info(f"Converted target column '{self.target_column}' to numeric (original name preserved)")
            else:
                transformed_data[self.target_column] = pd.to_numeric(target_data, errors='coerce')
                if transformed_data[self.target_column].isnull().any():
                    logger.error("Target contains non-numeric values that couldn't be coerced")
                    raise ValueError("Invalid values in target column")
                logger.info(f"Converted target column '{self.target_column}' to numeric (original name preserved)")

        # Ensure target column is the last column
        if self.target_column and self.target_column in transformed_data.columns:
            cols = [col for col in transformed_data.columns if col != self.target_column] + [self.target_column]
            transformed_data = transformed_data[cols]

        return transformed_data

    def save(self, path: str) -> None:
        """Save the preprocessing pipeline using cloudpickle"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            cloudpickle.dump(self, f)
        logger.info(f"Saved preprocessing pipeline to {path}")

    @classmethod
    def load(cls, path: str):
        """Load a preprocessing pipeline from a file"""
        with open(path, 'rb') as f:
            pipeline = cloudpickle.load(f)
        logger.info(f"Loaded preprocessing pipeline from {path}")
        return pipeline


def validate_and_sanitize_parameters(params: PreprocessingParameters, feature_store: Dict) -> PreprocessingParameters:
    """Validate and sanitize preprocessing parameters against feature store"""
    validated = params.dict()

    # Validate missing values columns
    valid_missing = [col for col in validated['missing_values_columns']
                     if col in feature_store.get('contains_null', [])]
    validated['missing_values_columns'] = valid_missing

    # Validate outlier columns
    valid_outliers = [col for col in validated['outliers_columns']
                      if col in feature_store.get('contains_outliers', [])]
    validated['outliers_columns'] = valid_outliers

    # Validate skewness columns
    valid_skewness = [col for col in validated['skewness_columns']
                      if col in feature_store.get('skewed_cols', [])]
    validated['skewness_columns'] = valid_skewness

    # Validate text columns
    valid_text = [col for col in validated['text_columns']
                  if col in feature_store.get('textual_cols', [])]
    validated['text_columns'] = valid_text

    return PreprocessingParameters(**validated)


async def api_preprocessing_workflow(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: PreprocessingParameters,
        config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    API-friendly preprocessing workflow

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        params: Preprocessing parameters
        config: Application config from intel.yaml

    Returns:
        Tuple of (train_preprocessed, test_preprocessed, preprocessing_config)
    """
    try:
        section("API PREPROCESSING WORKFLOW", logger)

        # Load feature store
        feature_store_path = config.get('feature_store_path')
        feature_store = load_yaml(feature_store_path)

        # Validate parameters against feature store
        validated_params = validate_and_sanitize_parameters(params, feature_store)

        # Initialize pipeline
        pipeline = PreprocessingPipeline(config, validated_params)
        pipeline.configure_pipeline()

        # Fit and transform
        section("FITTING PIPELINE", logger)
        pipeline.fit(train_df)

        section("TRANSFORMING DATA", logger)
        train_preprocessed = pipeline.transform(
            train_df,
            handle_duplicates=validated_params.handle_duplicates
        )
        test_preprocessed = pipeline.transform(
            test_df,
            handle_duplicates=False  # Never drop duplicates from test data
        )

        # Collect preprocessing config for response
        preprocessing_config = {
            'missing_values': {
                'method': validated_params.missing_values_method,
                'columns': validated_params.missing_values_columns
            },
            'outliers': {
                'method': validated_params.outliers_method,
                'columns': validated_params.outliers_columns
            },
            'skewness': {
                'method': validated_params.skewness_method,
                'columns': validated_params.skewness_columns
            },
            'scaling': {
                'method': validated_params.scaling_method,
                'columns': validated_params.scaling_columns
            },
            'categorical_encoding': {
                'method': validated_params.categorical_encoding_method,
                'columns': validated_params.categorical_columns,
                'drop_first': validated_params.drop_first
            },
            'text_preprocessing': {
                'enabled': validated_params.text_preprocessing_enabled,
                'columns': validated_params.text_columns
            }
        }

        # Move target column to last position if present
        target_column = config.get('target_col')
        if target_column in train_preprocessed.columns:
            cols = [col for col in train_preprocessed.columns if col != target_column] + [target_column]
            train_preprocessed = train_preprocessed[cols]
            test_preprocessed = test_preprocessed[cols]

        return train_preprocessed, test_preprocessed, preprocessing_config

    except Exception as e:
        logger.error(f"API Preprocessing failed: {str(e)}")
        raise


def save_preprocessing_artifacts(
        train_preprocessed: pd.DataFrame,
        test_preprocessed: pd.DataFrame,
        pipeline: PreprocessingPipeline,
        config: Dict
):
    """Save preprocessing results and pipeline"""
    try:
        # Create output directories
        interim_dir = Path(config.get('interim_dir', 'data/interim'))
        interim_dir.mkdir(parents=True, exist_ok=True)

        pipeline_dir = Path(config.get('pipeline_dir', 'model/pipelines'))
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessed data
        train_preprocessed_path = interim_dir / 'train_preprocessed.csv'
        test_preprocessed_path = interim_dir / 'test_preprocessed.csv'
        train_preprocessed.to_csv(train_preprocessed_path, index=False)
        test_preprocessed.to_csv(test_preprocessed_path, index=False)

        # Save pipeline
        pipeline_path = pipeline_dir / 'preprocessing.pkl'
        pipeline.save(pipeline_path)

        # Update config
        config.update({
            'train_preprocessed_path': str(train_preprocessed_path),
            'test_preprocessed_path': str(test_preprocessed_path),
            'preprocessing_pipeline_path': str(pipeline_path),
            'preprocessed_timestamp': datetime.now().isoformat()
        })

        return config

    except Exception as e:
        logger.error(f"Failed to save preprocessing artifacts: {str(e)}")
        raise


def load_yaml(file_path: str) -> Dict:
    """
    Load YAML file into a dictionary.

    Args:
        file_path (str): Path to YAML file

    Returns:
        Dict: Loaded YAML content
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {str(e)}")
        raise

def get_intel_config():
    """Safely load intel.yaml with validation"""
    try:
        with open('intel.yaml', 'r') as f:
            config = yaml.safe_load(f)
            # Validate critical keys
            if not config or 'dataset_name' not in config:
                raise ValueError("intel.yaml is missing required keys")
            return config
    except FileNotFoundError:
        raise RuntimeError("intel.yaml not found. Complete data ingestion first!")
    except Exception as e:
        raise RuntimeError(f"Invalid intel.yaml: {str(e)}")


def update_intel_yaml(intel_path: str, updates: Dict) -> None:
    """
    Update the intel.yaml file with new information.

    Args:
        intel_path (str): Path to intel.yaml file
        updates (Dict): Dictionary of updates to apply
    """
    try:
        # Load existing intel
        intel = load_yaml(intel_path)

        # Update with new information
        intel.update(updates)

        # Add processed timestamp
        intel['processed_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write back to file
        with open(intel_path, 'w') as file:
            yaml.dump(intel, file, default_flow_style=False)

        logger.info(f"Updated intel.yaml at {intel_path}")
    except Exception as e:
        logger.error(f"Error updating intel.yaml: {str(e)}")
        raise


def check_for_duplicates(df: pd.DataFrame) -> bool:
    """
    Check if dataframe contains duplicate rows.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        bool: True if duplicates exist, False otherwise
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows")
        return True
    else:
        logger.info("No duplicate rows found")
        return False


def check_for_skewness(df: pd.DataFrame, columns: List[str], threshold: float = 0.5) -> Dict[str, float]:
    """
    Check for skewness in the specified columns.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to check for skewness (should already exclude target)
        threshold (float): Skewness threshold (abs value) to consider a column skewed

    Returns:
        Dict[str, float]: Dictionary with column names as keys and skewness values as values
    """
    skewed_columns = {}

    for col in columns:
        if col not in df.columns:
            continue

        # Only process numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        skewness = df[col].skew()
        if abs(skewness) > threshold:
            skewed_columns[col] = skewness
            logger.info(f"Column {col} is skewed with skewness value: {skewness:.4f}")

    return skewed_columns


def get_numerical_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Get list of numerical columns in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude (List[str]): Columns to exclude

    Returns:
        List[str]: List of numerical columns
    """
    if exclude is None:
        exclude = []

    # Get columns with numeric dtype
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude specified columns
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    return numeric_cols


def get_categorical_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Get list of categorical columns in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude (List[str]): Columns to exclude

    Returns:
        List[str]: List of categorical columns
    """
    if exclude is None:
        exclude = []

    # Get columns with object or category dtype
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude specified columns
    cat_cols = [col for col in cat_cols if col not in exclude]

    return cat_cols


def recommend_skewness_transformer(df: pd.DataFrame, column: str) -> str:
    """
    Recommend the best transformer for a skewed column.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check

    Returns:
        str: Recommended transformer ('yeo-johnson' or 'box-cox')
    """
    # Safety check - don't process target column
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Column {column} is not numeric, defaulting to yeo-johnson")
        return 'yeo-johnson'

    # Check if column contains negative or zero values
    if df[column].min() <= 0:
        logger.info(f"Column {column} contains negative or zero values, recommending Yeo-Johnson transformation")
        return 'yeo-johnson'

    # Check the skewness after both transformations
    # Create a sample to test transformations (for speed)
    sample = df[column].sample(min(1000, len(df))).copy()

    # Test Yeo-Johnson
    try:
        yj_transformer = PowerTransformer(method='yeo-johnson')
        yj_transformed = yj_transformer.fit_transform(sample.values.reshape(-1, 1)).flatten()
        yj_skewness = stats.skew(yj_transformed)
    except Exception as e:
        logger.warning(f"Error testing Yeo-Johnson transformation: {str(e)}")
        yj_skewness = float('inf')

    # Test Box-Cox
    try:
        bc_transformer = PowerTransformer(method='box-cox')
        bc_transformed = bc_transformer.fit_transform(sample.values.reshape(-1, 1)).flatten()
        bc_skewness = stats.skew(bc_transformed)
    except Exception as e:
        logger.warning(f"Error testing Box-Cox transformation: {str(e)}")
        bc_skewness = float('inf')

    # Compare and recommend
    if abs(bc_skewness) <= abs(yj_skewness):
        logger.info(
            f"Box-Cox transformation recommended for {column} (skewness: {bc_skewness:.4f} vs {yj_skewness:.4f})")
        return 'box-cox'
    else:
        logger.info(f"Yeo-Johnson transformation recommended for {column} (skewness: {yj_skewness:.4f} vs {bc_skewness:.4f})")
        return 'yeo-johnson'


def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    try:
        section("DATA PREPROCESSING", logger)
        logger.info("Starting data preprocessing")

        # Load intel.yaml
        intel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'intel.yaml')
        intel = load_yaml(intel_path)
        logger.info(f"Loaded intel from {intel_path}")

        # Extract information from intel.yaml
        dataset_name = intel.get('dataset_name')
        feature_store_path = intel.get('feature_store_path')
        train_path = intel.get('cleaned_train_path')
        test_path = intel.get('cleaned_test_path')
        target_column = intel.get('target_col')

        # Load feature store
        feature_store = load_yaml(feature_store_path)
        logger.info(f"Loaded feature store from {feature_store_path}")

        # Check for special columns in feature store and exclude target column
        null_columns = [col for col in feature_store.get('contains_null', []) if col != target_column]
        outlier_columns = [col for col in feature_store.get('contains_outliers', []) if col != target_column]
        skewed_columns = [col for col in feature_store.get('skewed_cols', []) if col != target_column]
        categorical_columns = [col for col in feature_store.get('categorical_cols', []) if col != target_column]
        textual_columns = [col for col in feature_store.get('textual_cols', []) if col != target_column]

        # Load training and test data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded training data: {train_df.shape} and test data: {test_df.shape}")

        # Initialize preprocessing pipeline
        pipeline_config = {
            'dataset_name': dataset_name,
            'target_col': target_column,
            'feature_store': feature_store
        }
        pipeline = PreprocessingPipeline(pipeline_config)

        # Check for duplicates in training data
        has_duplicates = check_for_duplicates(train_df)

        # If categorical columns not specified in feature store, detect them automatically
        if not categorical_columns:
            categorical_columns = get_categorical_columns(train_df, exclude=[target_column])
            logger.info(f"Auto-detected categorical columns: {categorical_columns}")

        # If skewed columns not specified in feature store, detect them automatically
        if not skewed_columns:
            numerical_columns = get_numerical_columns(train_df, exclude=[target_column])
            skewness_dict = check_for_skewness(train_df, numerical_columns)
            skewed_columns = list(skewness_dict.keys())
            logger.info(f"Auto-detected skewed columns: {skewed_columns}")

        # Set up paths for output files
        interim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim',
                                   f'data_{dataset_name}')
        os.makedirs(interim_dir, exist_ok=True)

        train_preprocessed_path = os.path.join(interim_dir, 'train_preprocessed.csv')
        test_preprocessed_path = os.path.join(interim_dir, 'test_preprocessed.csv')

        pipeline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'pipelines',
                                    f'preprocessing_{dataset_name}')
        os.makedirs(pipeline_dir, exist_ok=True)

        pipeline_path = os.path.join(pipeline_dir, 'preprocessing.pkl')

        # Interactive preprocessing
        handle_duplicates = True  # Default is to handle duplicates

        # Handle missing values if any
        if null_columns:
            logger.info(f"Found columns with null values: {null_columns}")
            print(f"Found columns with null values: {null_columns}")
            print("How would you like to handle missing values?")
            print("1. Use mean (for numerical columns)")
            print("2. Use median (for numerical columns)")
            print("3. Use mode (most frequent value)")
            print("4. Drop rows with missing values")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                pipeline.handle_missing_values(method='mean', columns=null_columns)
            elif choice == '2':
                pipeline.handle_missing_values(method='median', columns=null_columns)
            elif choice == '3':
                pipeline.handle_missing_values(method='mode', columns=null_columns)
            elif choice == '4':
                pipeline.handle_missing_values(method='drop', columns=null_columns)
            else:
                logger.warning("Invalid choice, using default (mean)")
                pipeline.handle_missing_values(method='mean', columns=null_columns)

        # Handle duplicates if any
        if has_duplicates:
            print("Found duplicate rows in the training data.")
            print("How would you like to handle duplicates?")
            print("1. Drop duplicates")
            print("2. Keep duplicates")

            choice = input("Enter your choice (1-2): ")

            if choice == '1':
                handle_duplicates = True
            elif choice == '2':
                handle_duplicates = False
            else:
                logger.warning("Invalid choice, using default (drop duplicates)")
                handle_duplicates = True

        # Handle outliers if any
        if outlier_columns:
            logger.info(f"Found columns with outliers: {outlier_columns}")
            print(f"Found columns with outliers: {outlier_columns}")
            print("How would you like to handle outliers?")
            print("1. Use IQR method (cap at Q1 - 1.5 * IQR and Q3 + 1.5 * IQR)")
            print("2. Use Z-Score method (cap at mean ± 3 standard deviations)")

            choice = input("Enter your choice (1-2): ")

            if choice == '1':
                pipeline.handle_outliers(method='IQR', columns=outlier_columns)
            elif choice == '2':
                pipeline.handle_outliers(method='Z-Score', columns=outlier_columns)
            else:
                logger.warning("Invalid choice, using default (IQR)")
                pipeline.handle_outliers(method='IQR', columns=outlier_columns)

        # Handle skewed data if any
        if skewed_columns:
            logger.info(f"Found columns with skewed distributions: {skewed_columns}")
            print(f"Found columns with skewed distributions: {skewed_columns}")

            # Show recommended transformer for each skewed column
            print("\nRecommended transformers for each skewed column:")
            recommended_transformers = {}
            for col in skewed_columns:
                recommended = recommend_skewness_transformer(train_df, col)
                recommended_transformers[col] = recommended
                print(f"  - {col}: {recommended}")

            print("\nHow would you like to handle skewed data?")
            print("1. Use Yeo-Johnson transformation (works with negative values)")
            print("2. Use Box-Cox transformation (requires positive values)")
            print("3. Use recommended transformer for each column")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)
            elif choice == '2':
                pipeline.handle_skewed_data(method='box-cox', columns=skewed_columns)
            elif choice == '3':
                # We'll use the recommended transformer
                counts = {'yeo-johnson': 0, 'box-cox': 0}
                for col, transformer in recommended_transformers.items():
                    counts[transformer] += 1

                if counts['box-cox'] > counts['yeo-johnson']:
                    pipeline.handle_skewed_data(method='box-cox', columns=skewed_columns)
                else:
                    pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)
            else:
                logger.warning("Invalid choice, using default (Yeo-Johnson)")
                pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)

        # Scale numerical features
        numerical_columns = get_numerical_columns(train_df, exclude=[target_column])
        if numerical_columns:
            logger.info(f"Found numerical columns: {numerical_columns}")
            print(f"\nFound numerical columns: {numerical_columns}")
            print("Would you like to scale these numerical features?")
            print("1. Yes, use StandardScaler (mean=0, std=1)")
            print("2. Yes, use RobustScaler (median=0, IQR=1, robust to outliers)")
            print("3. Yes, use MinMaxScaler (scale to range [0,1])")
            print("4. No, do not scale numerical features")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                pipeline.scale_numerical_features(method='standard', columns=numerical_columns)
            elif choice == '2':
                pipeline.scale_numerical_features(method='robust', columns=numerical_columns)
            elif choice == '3':
                pipeline.scale_numerical_features(method='minmax', columns=numerical_columns)
            elif choice == '4':
                logger.info("Skipping numerical feature scaling")
            else:
                logger.warning("Invalid choice, skipping numerical feature scaling")

        # Encode categorical features
        if categorical_columns:
            logger.info(f"Found categorical columns: {categorical_columns}")
            print(f"\nFound categorical columns: {categorical_columns}")
            print("How would you like to encode these categorical features?")
            print("1. Use OneHotEncoder (sklearn)")
            print("2. Use pd.get_dummies (pandas)")
            print("3. Use LabelEncoder (convert to integers)")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                drop_first = input("Drop first category to avoid multicollinearity? (y/n): ").lower() == 'y'
                pipeline.encode_categorical_features(method='onehot', columns=categorical_columns, drop_first=drop_first)
            elif choice == '2':
                drop_first = input("Drop first category to avoid multicollinearity? (y/n): ").lower() == 'y'
                pipeline.encode_categorical_features(method='dummies', columns=categorical_columns, drop_first=drop_first)
            elif choice == '3':
                pipeline.encode_categorical_features(method='label', columns=categorical_columns)
            else:
                logger.warning("Invalid choice, using default (OneHotEncoder)")
                pipeline.encode_categorical_features(method='onehot', columns=categorical_columns)

            # Handle text preprocessing if textual columns exist
            if textual_columns:
                logger.info(f"Found textual columns: {textual_columns}")
                print(f"\nFound textual columns: {textual_columns}")
                print("Would you like to preprocess these text columns?")
                print("1. Yes, with basic preprocessing")
                print("2. Yes, with advanced preprocessing")
                print("3. No, keep text as is")

                choice = input("Enter your choice (1-3): ")

                if choice in ['1', '2']:
                    # Basic text preprocessing config
                    text_config = {
                        'lowercase': True,
                        'remove_html': True,
                        'remove_urls': True,
                        'handle_emojis': 'remove',
                        'remove_punctuation': True,
                        'handle_chat_words': False,
                        'spelling_correction': False,
                        'remove_stopwords': True,
                        'stemming_lemmatization': 'none',
                        'pos_tagging': False,
                        'tokenization_method': 'tfidf',
                        'ngram_range': [1, 2],
                        'max_features': 1000,
                        'vector_size': 100
                    }

                    if choice == '2':
                        # Advanced preprocessing options
                        print("\nAdvanced Text Preprocessing Options:")

                        # Stemming/Lemmatization
                        print("Choose stemming/lemmatization:")
                        print("1. None")
                        print("2. Stemming (Porter)")
                        print("3. Lemmatization")
                        stem_choice = input("Enter choice (1-3): ")
                        if stem_choice == '2':
                            text_config['stemming_lemmatization'] = 'stemming'
                        elif stem_choice == '3':
                            text_config['stemming_lemmatization'] = 'lemmatization'

                        # Tokenization method
                        print("\nChoose tokenization method:")
                        print("1. TF-IDF (default)")
                        print("2. Bag of Words")
                        print("3. Word2Vec")
                        print("4. N-grams")
                        token_choice = input("Enter choice (1-4): ")
                        if token_choice == '2':
                            text_config['tokenization_method'] = 'bow'
                        elif token_choice == '3':
                            text_config['tokenization_method'] = 'word2vec'
                        elif token_choice == '4':
                            text_config['tokenization_method'] = 'ngrams'

                        try:
                            max_feat = int(input("Maximum features to extract (default 1000): ") or "1000")
                            text_config['max_features'] = max_feat
                        except:
                            pass

                    pipeline.setup_text_preprocessing(text_config, textual_columns)
                else:
                    logger.info("Skipping text preprocessing")

            # Collect preprocessing configuration from the pipeline
            preprocessing_config = {}

            if pipeline.missing_handler:
                preprocessing_config['missing_values'] = {
                    'method': pipeline.missing_handler.method,
                    'columns': pipeline.missing_handler.columns
                }

            if pipeline.outlier_handler:
                preprocessing_config['outliers'] = {
                    'method': pipeline.outlier_handler.method,
                    'columns': pipeline.outlier_handler.columns
                }

            if pipeline.skewed_handler:
                preprocessing_config['skewed_data'] = {
                    'method': pipeline.skewed_handler.method,
                    'columns': pipeline.skewed_handler.columns
                }

            if pipeline.numerical_scaler:
                preprocessing_config['numerical_scaling'] = {
                    'method': pipeline.numerical_scaler.method,
                    'columns': pipeline.numerical_scaler.columns
                }

            if pipeline.categorical_encoder:
                preprocessing_config['categorical_encoding'] = {
                    'method': pipeline.categorical_encoder.method,
                    'columns': pipeline.categorical_encoder.columns,
                    'drop_first': pipeline.categorical_encoder.drop_first
                }

            if pipeline.text_preprocessor:
                preprocessing_config['text_preprocessing'] = {
                    'enabled': True,
                    'columns': pipeline.text_preprocessor.columns,
                    'tokenization_method': pipeline.text_preprocessor.tokenization_method
                }

            preprocessing_config['handle_duplicates'] = handle_duplicates

            section("FITTING PIPELINE", logger)
            pipeline.fit(train_df)

            section("TRANSFORMING DATA", logger)
            train_preprocessed = pipeline.transform(train_df, handle_duplicates=handle_duplicates)
            if target_column in train_preprocessed.columns:
                cols = [col for col in train_preprocessed.columns if col != target_column] + [target_column]
                train_preprocessed = train_preprocessed[cols]
            logger.info(f"Transformed training data: {train_preprocessed.shape}")

            test_preprocessed = pipeline.transform(test_df, handle_duplicates=False)
            if target_column in test_preprocessed.columns:
                cols = [col for col in test_preprocessed.columns if col != target_column] + [target_column]
                test_preprocessed = test_preprocessed[cols]
            logger.info(f"Transformed test data: {test_preprocessed.shape}")

            # Save preprocessing artifacts
            try:
                interim_dir = Path('data/interim')
                interim_dir.mkdir(parents=True, exist_ok=True)

                pipeline_dir = Path('model/pipelines')
                pipeline_dir.mkdir(parents=True, exist_ok=True)

                train_preprocessed.to_csv(interim_dir / 'train_preprocessed.csv', index=False)
                test_preprocessed.to_csv(interim_dir / 'test_preprocessed.csv', index=False)
                logger.info("Saved preprocessed data to data/interim/")

                pipeline_path = pipeline_dir / 'preprocessing.pkl'
                pipeline.save(pipeline_path)
                logger.info(f"Saved preprocessing pipeline to {pipeline_path}")

                intel_updates = {
                    'train_preprocessed_path': str(interim_dir / 'train_preprocessed.csv'),
                    'test_preprocessed_path': str(interim_dir / 'test_preprocessed.csv'),
                    'preprocessing_pipeline_path': str(pipeline_path),
                    'preprocessing_config': preprocessing_config,
                    'preprocessed_timestamp': datetime.now().isoformat()
                }
                update_intel_yaml(intel_path, intel_updates)

                logger.info("Data preprocessing completed successfully!")
                section("PREPROCESSING COMPLETE", logger)

            except Exception as e:
                logger.error(f"Error saving preprocessing artifacts: {str(e)}")
                raise

        else:
            logger.error("No preprocessing steps were configured. Please check your feature store configuration.")
            raise ValueError("No preprocessing steps configured")

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
