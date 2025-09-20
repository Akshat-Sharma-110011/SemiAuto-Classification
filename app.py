from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from io import BytesIO
import pandas as pd
import os
import yaml
import tempfile
from pathlib import Path
import uvicorn
import shutil
import sys

# Configure logging first thing - before any other imports that might use logging
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger("FastAPI-App")

# Import internal modules after logging is configured
try:
    from semiauto_classification.features.feature_selection import feature_selection_pipeline, \
        MetaHeuristicFeatureSelector
    from semiauto_classification.data.data_ingestion import create_data_ingestion
    from semiauto_classification.utils import load_yaml, update_intel_yaml
    from semiauto_classification.data.data_cleaning import main as data_cleaning_main
    from semiauto_classification.features.feature_engineering import run_feature_engineering
    from semiauto_classification.model.model_building import (
        build_single_model_api_patched,
        build_parallel_models_api_patched,
        build_ensemble_model_api,
        select_best_model_api_patched,
        EnhancedModelBuilder,
        build_staged_ensemble_api
    )
    from semiauto_classification.model.model_evaluation import run_evaluation, get_evaluation_summary
    from semiauto_classification.model.model_optimization import optimize_model
    from semiauto_classification.data.data_preprocessing import (
        PreprocessingPipeline, PreprocessingParameters, ImagePreprocessingConfig,
        check_for_duplicates, get_numerical_columns,
        get_categorical_columns, recommend_skewness_transformer,
        preprocess_image_dataset
    )

    # Set a flag to indicate successful imports
    MODULES_AVAILABLE = True
    logger.info("All internal modules imported successfully")

except ImportError as e:
    logger.error(f"Some internal modules could not be imported: {str(e)}")
    MODULES_AVAILABLE = False

    # Create placeholder functions to prevent crashes
    def placeholder_function(*args, **kwargs):
        logger.error("Internal module function called but not available")
        raise HTTPException(status_code=500, detail=f"Required module not available. Please install missing dependencies: {str(e)}")

    # Create placeholder classes
    class PreprocessingPipeline:
        def __init__(self, *args, **kwargs):
            pass

    class PreprocessingParameters:
        def __init__(self, *args, **kwargs):
            pass

    class ImagePreprocessingConfig:
        def __init__(self, *args, **kwargs):
            pass

    # Assign placeholder functions
    feature_selection_pipeline = placeholder_function
    MetaHeuristicFeatureSelector = type('MetaHeuristicFeatureSelector', (), {})
    create_data_ingestion = placeholder_function
    load_yaml = placeholder_function
    update_intel_yaml = placeholder_function
    data_cleaning_main = placeholder_function
    run_feature_engineering = placeholder_function
    build_single_model_api_patched = placeholder_function
    build_parallel_models_api_patched = placeholder_function
    build_ensemble_model_api = placeholder_function
    select_best_model_api_patched = placeholder_function
    EnhancedModelBuilder = type('EnhancedModelBuilder', (), {})
    build_staged_ensemble_api = placeholder_function
    run_evaluation = placeholder_function
    get_evaluation_summary = placeholder_function
    optimize_model = placeholder_function
    check_for_duplicates = placeholder_function
    get_numerical_columns = placeholder_function
    get_categorical_columns = placeholder_function
    recommend_skewness_transformer = placeholder_function
    preprocess_image_dataset = placeholder_function

# Log application startup
logger.info("Starting SemiAuto Classification FastAPI application")

app = FastAPI(title="SemiAuto Classification", version="1.0")

# Set up static files and templates
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/classification-static", StaticFiles(directory=static_dir), name="classification-static")

templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)
templates.env.filters["zip"] = zip

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

logger.info("FastAPI application and middleware configured")


# Add a health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify module availability"""
    return {
        "status": "healthy" if MODULES_AVAILABLE else "degraded",
        "modules_available": MODULES_AVAILABLE,
        "message": "All modules loaded successfully" if MODULES_AVAILABLE else "Some modules are missing - please install dependencies"
    }


class ProcessingModeConfig(BaseModel):
    """Configuration for processing mode selection"""
    mode: str = Field(..., description="Processing mode: 'auto', 'tabular', 'textual', 'image', or 'mixed'")


class TextPreprocessingConfig(BaseModel):
    """Configuration for text preprocessing options"""
    lowercase: bool = False
    remove_html: bool = False
    remove_urls: bool = False
    handle_emojis: str = "keep"  # keep, remove, replace
    remove_punctuation: bool = False
    handle_chat_words: bool = False
    spelling_correction: bool = False
    remove_stopwords: bool = False
    stemming_lemmatization: str = "none"  # none, stemming, lemmatization
    pos_tagging: bool = False
    tokenization_method: str = "none"  # none, bow, tfidf, word2vec, glove, ngrams
    ngram_range: Optional[List[int]] = [1, 2]  # for n-grams
    max_features: Optional[int] = 1000  # for vectorization methods
    vector_size: Optional[int] = 100  # for word2vec/glove


class ImagePreprocessingConfig(BaseModel):
    """Configuration for image preprocessing options"""
    resize_images: bool = True
    target_size: List[int] = [224, 224]  # width, height
    normalize: bool = True
    augmentation: bool = False
    augmentation_options: Optional[Dict[str, Any]] = {
        'rotation': 10,
        'zoom': 0.1,
        'horizontal_flip': True,
        'vertical_flip': False
    }


class EnhancedImagePreprocessingConfig(BaseModel):
    """Enhanced configuration for image preprocessing options"""
    # Resizing/Rescaling
    resize_images: bool = True
    target_size: List[int] = [224, 224]  # width, height
    maintain_aspect_ratio: bool = False
    padding_color: List[int] = [0, 0, 0]  # RGB values for padding

    # Color Normalization
    normalize: bool = True
    normalization_method: str = 'standard'  # standard, minmax, custom
    mean: List[float] = [0.485, 0.456, 0.406]  # ImageNet means
    std: List[float] = [0.229, 0.224, 0.225]  # ImageNet stds

    # Noise Reduction/Smoothing
    noise_reduction: bool = False
    noise_method: str = 'gaussian'  # gaussian, median, bilateral
    kernel_size: int = 5

    # Image Enhancement
    enhance_images: bool = False
    enhance_brightness: float = 1.0
    enhance_contrast: float = 1.0
    enhance_sharpness: float = 1.0
    enhance_color: float = 1.0

    # Segmentation/ROI Extraction
    roi_extraction: bool = False
    roi_method: str = 'contour'  # contour, threshold, watershed
    roi_threshold: int = 127

    # Morphological Operations
    morphological_ops: bool = False
    morph_operation: str = 'opening'  # opening, closing, gradient, tophat, blackhat
    morph_kernel_size: int = 5
    morph_iterations: int = 1

    # Binarization/Thresholding
    binarization: bool = False
    threshold_method: str = 'otsu'  # otsu, adaptive, binary, truncate
    threshold_value: int = 127

    # Data Augmentation
    augmentation: bool = False
    augmentation_factor: float = 1.0  # multiplier for augmented images
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: float = 30.0
    zoom_range: float = 0.2
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    blur_limit: int = 3
    random_crop: bool = False
    elastic_transform: bool = False
    noise_augmentation: bool = False


class PreprocessingConfig(BaseModel):
    missing_values: Optional[str] = None
    handle_duplicates: bool = True
    outliers: Optional[str] = None
    skewedness: Optional[str] = None
    scaling: Optional[str] = None
    encoding: Optional[str] = None
    drop_first: Optional[bool] = False
    text_preprocessing: Optional[TextPreprocessingConfig] = Field(
        default_factory=lambda: TextPreprocessingConfig(
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            handle_emojis='remove',
            remove_punctuation=True,
            remove_stopwords=True,
            tokenization_method='tfidf',
            ngram_range=[1, 2],
            max_features=1000
        )
    )
    image_preprocessing: Optional[EnhancedImagePreprocessingConfig] = Field(
        default_factory=lambda: EnhancedImagePreprocessingConfig(
            resize_images=True,
            target_size=[224, 224],
            normalize=True,
            augmentation=False
        )
    )


# Function to check module availability before calling functions
def check_modules_available():
    if not MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Required modules are not available. Please install missing dependencies. Run: pip install albumentations opencv-python"
        )


class FeatureEngineeringRequest(BaseModel):
    use_feature_tools: bool = False
    use_shap: bool = False
    n_features: int = 20


class ModelBuildRequest(BaseModel):
    model_name: str
    custom_params: Optional[Dict[str, Any]] = None


class ParallelModelRequest(BaseModel):
    models: List[Dict[str, Any]]
    max_workers: int = 4


class SimpleEnsembleRequest(BaseModel):
    ensemble_type: str  # voting, stacking, bagging
    config: Dict[str, Any]


class MultiStageStackingRequest(BaseModel):
    stages: List[Dict[str, Any]]
    final_estimator: Optional[Dict[str, Any]] = None
    cv: int = 5
    passthrough: bool = False


class HybridEnsembleRequest(BaseModel):
    ensemble_type: str  # voting_stacking, bagging_stacking
    config: Dict[str, Any]


class EnsembleModelRequest(BaseModel):
    ensemble_category: str = "simple"  # simple, multi_stage, hybrid
    ensemble_type: str  # voting, stacking, bagging, etc.
    config: Dict[str, Any] = {}


class StagedEnsembleRequest(BaseModel):
    architecture: List[Dict[str, Any]]  # Each stage configuration
    final_meta_model: Optional[Dict[str, Any]] = None
    cv: int = 5
    stage_method: str = "stacking"  # stacking, voting, or mixed
    combine_method: str = "concatenate"  # concatenate, average, weighted_average
    passthrough_original: bool = False


class ModelSelectionRequest(BaseModel):
    parallel_results: Dict[str, Any]
    selection_metric: str = "accuracy"


class OptimizationRequest(BaseModel):
    optimize: bool = True
    method: str = "1"  # "1"=GridSearch, "2"=Optuna
    n_trials: int = 50
    metric: str = "1"  # e.g., "1"=Accuracy


# Route to main page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Serving main page")
    return templates.TemplateResponse("index.html", {"request": request})


# Route to data upload page
@app.get("/data-upload", response_class=HTMLResponse)
async def data_upload_page(request: Request):
    logger.info("Serving data upload page")
    return templates.TemplateResponse("data_upload.html", {"request": request, "modules_available": MODULES_AVAILABLE})


# Route to preprocessing page with enhanced image support
@app.get("/preprocessing", response_class=HTMLResponse)
async def preprocessing_page(request: Request):
    logger.info("Serving preprocessing page")
    try:
        check_modules_available()
        intel = load_yaml("intel.yaml")
        feature_store = {}
        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])

        logger.info(f"Loaded preprocessing page data for dataset: {intel.get('dataset_name', 'Unknown')}")

        processing_mode = feature_store.get('processing_mode', 'tabular')

        template_data = {
            "request": request,
            "dataset_name": intel.get('dataset_name', ''),
            "target_column": intel.get('target_column', ''),
            "processing_mode": processing_mode,
            "numerical_cols": feature_store.get('numerical_cols', []),
            "categorical_cols": feature_store.get('categorical_cols', []),
            "textual_cols": feature_store.get('textual_cols', []),
            "image_cols": feature_store.get('image_cols', []),
            "nulls": feature_store.get('contains_null', []),
            "outliers": feature_store.get('contains_outliers', []),
            "skewed": feature_store.get('skewed_cols', [])
        }

        if 'text_analysis' in feature_store:
            template_data['text_analysis'] = feature_store['text_analysis']

        if 'image_analysis' in feature_store:
            template_data['image_analysis'] = feature_store['image_analysis']

        return templates.TemplateResponse("preprocessing.html", template_data)
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.warning(f"Failed to load preprocessing page data: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please upload data first",
            "redirect_url": "/data-upload"
        })


# Route to feature engineering page
@app.get("/feature-engineering", response_class=HTMLResponse)
async def feature_engineering_page(request: Request):
    logger.info("Serving feature engineering page")
    try:
        check_modules_available()
        intel = load_yaml("intel.yaml")
        if not intel.get('train_preprocessed_path') and not intel.get('train_images_path'):
            logger.warning("Preprocessing not completed, redirecting to preprocessing page")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please complete preprocessing first",
                "redirect_url": "/preprocessing"
            })

        # Load feature store to get processing mode
        feature_store = {}
        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])

        processing_mode = feature_store.get('processing_mode', 'tabular')

        return templates.TemplateResponse("feature_engineering.html", {
            "request": request,
            "processing_mode": processing_mode,
            "textual_cols": feature_store.get('textual_cols', []),
            "image_cols": feature_store.get('image_cols', [])
        })
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Error loading feature engineering page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


@app.get("/feature-selection", response_class=HTMLResponse)
async def feature_selection_page(request: Request):
    logger.info("Serving feature selection page")
    try:
        check_modules_available()
        intel = load_yaml("intel.yaml")
        if not intel.get('train_transformed_path') and not intel.get('train_images_path'):
            logger.warning("Feature engineering not completed, redirecting")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please complete feature engineering first",
                "redirect_url": "/feature-engineering"
            })
        return templates.TemplateResponse("feature_selection.html", {"request": request})
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Error loading feature selection page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# Route to model building page
@app.get("/model-building", response_class=HTMLResponse)
async def model_building_page(request: Request):
    logger.info("Serving model building page")
    try:
        check_modules_available()
        intel = load_yaml("intel.yaml")
        if not intel.get('train_selected_path') and not intel.get('train_transformed_path') and not intel.get(
                'train_images_path'):
            logger.warning("Feature engineering not completed, redirecting to feature engineering page")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please complete feature engineering first",
                "redirect_url": "/feature-engineering"
            })

        builder = EnhancedModelBuilder()
        available_models = builder.get_available_models()
        logger.info(f"Loaded {len(available_models)} available models")

        # Load processing mode information
        feature_store = {}
        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])

        processing_mode = feature_store.get('processing_mode', 'tabular')

        return templates.TemplateResponse("model_building.html", {
            "request": request,
            "available_models": list(available_models.keys()),
            "processing_mode": processing_mode
        })
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Error loading model building page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# Route to model optimization page
@app.get("/optimization", response_class=HTMLResponse)
async def optimization_page(request: Request):
    logger.info("Serving optimization page")
    try:
        check_modules_available()
        intel = load_yaml("intel.yaml")
        if not intel.get('model_path'):
            logger.warning("Model not built, redirecting to model building page")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please build a model first",
                "redirect_url": "/model-building"
            })

        return templates.TemplateResponse("optimization.html", {"request": request})
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Error loading optimization page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# Route to results page
@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    logger.info("Serving results page")
    try:
        check_modules_available()
        intel = load_yaml("intel.yaml")
        if not intel.get('performance_metrics_path'):
            logger.warning("No evaluation results available, redirecting to model building page")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "No model evaluation results available",
                "redirect_url": "/model-building"
            })

        metrics = load_yaml(intel['performance_metrics_path'])
        logger.info(f"Loaded evaluation metrics for dataset: {intel.get('dataset_name', 'unnamed')}")

        # Load processing mode information
        feature_store = {}
        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])

        processing_mode = feature_store.get('processing_mode', 'tabular')

        return templates.TemplateResponse("results.html", {
            "request": request,
            "metrics": metrics,
            "dataset_name": intel.get('dataset_name', 'unnamed'),
            "processing_mode": processing_mode
        })
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Error loading results page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# API endpoints
@app.post("/api/upload")
async def upload_dataset(
        file: Optional[UploadFile] = File(None),
        target_column: str = Form(...),
        processing_mode: str = Form("auto"),
        images_zip: Optional[UploadFile] = File(None),
        labels_csv: Optional[UploadFile] = File(None),
):
    logger.info(
        f"Received upload request - processing mode: {processing_mode}, target column: {target_column}"
    )

    # Check if modules are available
    check_modules_available()

    # Handle image mode
    if processing_mode == "image" or (images_zip and labels_csv):
        if not (images_zip and labels_csv):
            raise HTTPException(
                status_code=400,
                detail="Images zip and labels CSV are required for image mode",
            )

        logger.info(
            f"Image mode upload: images={images_zip.filename}, labels={labels_csv.filename}"
        )

        # Validate files
        if not images_zip.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Images must be a zip file")
        if not labels_csv.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Labels must be a CSV file")

        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_images:
                images_content = await images_zip.read()
                temp_images.write(images_content)
                temp_images.flush()
                images_path = temp_images.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_labels:
                labels_content = await labels_csv.read()
                temp_labels.write(labels_content)
                temp_labels.flush()
                labels_path = temp_labels.name

            ingestion = create_data_ingestion()
            dataset_name = os.path.splitext(images_zip.filename)[0]

            result = ingestion.run_ingestion_pipeline(
                mode="image",
                images_zip=images_path,
                labels_csv=labels_path,
                filename=dataset_name,
                target_col=target_column,
            )

            # Clean up
            os.unlink(images_path)
            os.unlink(labels_path)

            ingestion.save_intel_yaml()
            logger.info("Image ingestion completed successfully")

            return {
                "message": "Image dataset ingestion completed",
                "processing_mode": result.get("processing_mode", "image"),
                "dataset_info": {
                    "total_images": result.get("total_images", 0),
                    "total_labels": result.get("total_labels", 0),
                    "classes": result.get("classes", []),
                    "image_formats": result.get("image_formats", {}),
                    "images_path": result.get("images_path"),
                    "labels_path": result.get("labels_path"),
                },
            }

        except Exception as e:
            # Clean up temp files if they exist
            for temp_path in [locals().get("images_path"), locals().get("labels_path")]:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            logger.error(f"Error during image upload: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error processing image files: {e}"
            )

    # Handle tabular/textual mode
    else:
        if not file:
            raise HTTPException(status_code=400, detail="CSV file is required for tabular/textual modes")

        if not file.filename.endswith(".csv"):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only CSV files are supported for tabular/textual modes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            contents = await file.read()
            temp.write(contents)
            temp.flush()
            path = temp.name

        try:
            df = pd.read_csv(path)
            columns = df.columns.tolist()
            logger.info(f"Successfully read CSV with {len(df)} rows and {len(columns)} columns")

            ingestion = create_data_ingestion()
            with open(path, "rb") as f:
                # Pass processing mode to ingestion pipeline
                mode = None if processing_mode == "auto" else processing_mode
                result = ingestion.run_ingestion_pipeline(f, file.filename, target_column, mode)

            os.unlink(path)
            ingestion.save_intel_yaml()
            logger.info(
                f"Data ingestion completed successfully with processing mode: {result.get('processing_mode', 'unknown')}")

            # Run data cleaning
            try:
                data_cleaning_main()
                logger.info("Data cleaning completed successfully")
            except Exception as e:
                logger.error(f"Data cleaning failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")

            return {
                "message": "Data ingestion and cleaning completed",
                "columns": columns,
                "processing_mode": result.get('processing_mode', 'tabular'),
                "dataset_info": {
                    "shape": result.get('data_shape'),
                    "numerical_columns": result.get('numerical_columns', []),
                    "categorical_columns": result.get('categorical_columns', []),
                    "textual_columns": result.get('textual_columns', [])
                }
            }
        except Exception as e:
            logger.error(f"Error during data upload: {str(e)}")
            if os.path.exists(path):
                os.unlink(path)
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/processing-modes")
async def get_processing_modes():
    """Get available processing modes and their descriptions"""
    return {
        "auto": "Automatically detect the best processing mode based on data characteristics",
        "tabular": "Traditional tabular data with numerical and categorical features",
        "textual": "Text-heavy data suitable for NLP tasks",
        "image": "Image datasets with images and corresponding labels",
        "mixed": "Data containing both tabular and textual features"
    }


@app.post("/api/set-processing-mode")
async def set_processing_mode(config: ProcessingModeConfig):
    """Manually set the processing mode for the current dataset"""
    check_modules_available()
    logger.info(f"Setting processing mode to: {config.mode}")

    try:
        # Load current intel
        intel = load_yaml("intel.yaml")

        # Load and update feature store
        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])
            feature_store['processing_mode'] = config.mode

            # Save updated feature store
            with open(intel['feature_store_path'], 'w') as f:
                yaml.dump(feature_store, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Processing mode updated to: {config.mode}")
            return {
                "status": "success",
                "message": f"Processing mode set to {config.mode}",
                "processing_mode": config.mode
            }
        else:
            raise HTTPException(status_code=404, detail="Feature store not found. Please upload data first.")

    except Exception as e:
        logger.error(f"Error setting processing mode: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set processing mode: {str(e)}")


@app.get("/api/dataset-info")
async def get_dataset_info():
    """Get information about the currently loaded dataset"""
    check_modules_available()
    try:
        intel = load_yaml("intel.yaml")
        feature_store = {}

        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])

        processing_mode = feature_store.get('processing_mode', 'tabular')

        # Base information common to all modes
        base_info = {
            "dataset_name": intel.get('dataset_name', 'Unknown'),
            "processing_mode": processing_mode,
            "target_column": intel.get('target_column', ''),
        }

        if processing_mode == 'image':
            # Image-specific information
            image_analysis = feature_store.get('image_analysis', {})
            base_info.update({
                "images_path": intel.get('images_path', ''),
                "labels_path": intel.get('labels_path', ''),
                "total_images": image_analysis.get('total_images', 0),
                "total_labels": image_analysis.get('total_labels', 0),
                "classes": image_analysis.get('classes', []),
                "image_formats": image_analysis.get('image_formats', {}),
                "label_format": image_analysis.get('label_format', 'unknown'),
                "avg_image_size": image_analysis.get('avg_image_size', None),
                "columns": {
                    "image": feature_store.get('image_cols', [])
                }
            })
        else:
            # Tabular/textual information
            base_info.update({
                "columns": {
                    "numerical": feature_store.get('numerical_cols', []),
                    "categorical": feature_store.get('categorical_cols', []),
                    "textual": feature_store.get('textual_cols', []),
                    "id": feature_store.get('id_cols', [])
                },
                "data_quality": {
                    "contains_null": feature_store.get('contains_null', []),
                    "contains_outliers": feature_store.get('contains_outliers', []),
                    "skewed_cols": feature_store.get('skewed_cols', [])
                },
                "text_analysis": feature_store.get('text_analysis', {})
            })

        return base_info

    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=404, detail="Dataset information not found")


@app.post("/api/preprocess")
def preprocess_data(config: PreprocessingConfig):
    logger.info("Starting data preprocessing")
    check_modules_available()

    try:
        intel = load_yaml("intel.yaml")
        feature_store = load_yaml(intel['feature_store_path'])
        processing_mode = feature_store.get('processing_mode', 'tabular')

        logger.info(f"Processing mode: {processing_mode}")

        if processing_mode == 'image':
            return preprocess_image_data(config, intel, feature_store)
        else:
            return preprocess_tabular_data(config, intel, feature_store)

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


def preprocess_image_data(config: PreprocessingConfig, intel: dict, feature_store: dict):
    """Handle image-specific preprocessing with comprehensive options - Fixed version"""
    logger.info("Starting image preprocessing")

    try:
        if not config.image_preprocessing:
            # Use default configuration
            image_config = EnhancedImagePreprocessingConfig()
        else:
            image_config = config.image_preprocessing

        # Convert to dictionary format for the preprocessing function
        preprocessing_config = image_config.dict()

        logger.info(f"Image preprocessing configuration: {preprocessing_config}")

        # Call the image preprocessing function
        result = preprocess_image_dataset("intel.yaml", preprocessing_config)

        # The result already contains all necessary paths from the fixed function
        return {
            "message": result['message'],
            "processing_mode": "image",
            "processed_images": {
                "original_train_images": result['original_train_images'],
                "processed_train_images": result['processed_train_images'],
                "original_test_images": result['original_test_images'],
                "processed_test_images": result['processed_test_images'],
                "augmentation_factor": result.get('augmentation_factor', 1.0),
                "preprocessing_applied": True
            },
            "paths": {
                "train_images": result.get('train_images_preprocessed_path'),
                "test_images": result.get('test_images_preprocessed_path'),
                "train_labels": result.get('train_labels_preprocessed_path'),
                "test_labels": result.get('test_labels_preprocessed_path')
            }
        }
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {str(e)}")


def preprocess_tabular_data(config: PreprocessingConfig, intel: dict, feature_store: dict):
    """Handle tabular/textual preprocessing"""
    try:
        train_df = pd.read_csv(intel['cleaned_train_path'])
        test_df = pd.read_csv(intel['cleaned_test_path']) if 'cleaned_test_path' in intel else pd.DataFrame()

        logger.info(f"Loaded data for preprocessing: {len(train_df)} train rows, {len(test_df)} test rows")

        numerical = feature_store.get('numerical_cols', [])
        categorical = feature_store.get('categorical_cols', [])
        textual = feature_store.get('textual_cols', [])
        nulls = [col for col in feature_store.get('contains_null', []) if col != intel['target_column']]
        outliers = [col for col in feature_store.get('contains_outliers', []) if col != intel['target_column']]
        skewed = [col for col in feature_store.get('skewed_cols', []) if col != intel['target_column']]

        # Create preprocessing parameters with text config
        params = PreprocessingParameters(
            text_preprocessing_enabled=config.text_preprocessing is not None,
            text_columns=textual,
            text_preprocessing_config=config.text_preprocessing.dict() if config.text_preprocessing else None,
            missing_values_method=config.missing_values or 'mean',
            missing_values_columns=nulls,
            handle_duplicates=config.handle_duplicates,
            outliers_method=config.outliers,
            outliers_columns=outliers,
            skewness_method=config.skewedness,
            skewness_columns=skewed,
            scaling_method=config.scaling,
            scaling_columns=numerical,
            categorical_encoding_method=config.encoding,
            categorical_columns=categorical,
            drop_first=config.drop_first or False
        )

        pipeline = PreprocessingPipeline({
            'dataset_name': intel['dataset_name'],
            'target_col': intel['target_column'],
            'feature_store': feature_store
        }, params)

        pipeline.configure_pipeline()

        pipeline.fit(train_df)
        train_p = pipeline.transform(train_df, handle_duplicates=config.handle_duplicates)
        test_p = pipeline.transform(test_df, handle_duplicates=False)

        interim_dir = Path(f"data/interim/data_{intel['dataset_name']}")
        interim_dir.mkdir(parents=True, exist_ok=True)
        train_path = interim_dir / "train_preprocessed.csv"
        test_path = interim_dir / "test_preprocessed.csv"

        pipeline_dir = Path(f"model/pipelines/preprocessing_{intel['dataset_name']}")
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = pipeline_dir / "preprocessing.pkl"

        train_p.to_csv(train_path, index=False)
        test_p.to_csv(test_path, index=False)
        pipeline.save(str(pipeline_path))

        update_intel_yaml("intel.yaml", {
            "train_preprocessed_path": str(train_path),
            "test_preprocessed_path": str(test_path),
            "preprocessing_pipeline_path": str(pipeline_path)
        })

        logger.info("Data preprocessing completed successfully")
        return {
            "message": "Preprocessing completed",
            "processing_mode": feature_store.get('processing_mode', 'tabular'),
            "processed_columns": {
                "numerical": len(numerical),
                "categorical": len(categorical),
                "textual": len(textual)
            }
        }

    except Exception as e:
        logger.error(f"Error during tabular preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tabular preprocessing failed: {str(e)}")


# API endpoint to get image preprocessing options
@app.get("/api/image-preprocessing-options")
async def get_image_preprocessing_options():
    """Get available image preprocessing options and their descriptions"""
    return {
        "resize_options": {
            "resize_images": "Resize all images to a standard size",
            "target_size": "Target dimensions [width, height] in pixels",
            "maintain_aspect_ratio": "Preserve original aspect ratio with padding",
            "padding_color": "RGB color values for padding [R, G, B]"
        },
        "normalization_options": {
            "normalize": "Apply color normalization to images",
            "normalization_method": "Method: 'standard' (ImageNet), 'minmax', 'custom'",
            "mean": "Mean values for standard normalization [R, G, B]",
            "std": "Standard deviation for normalization [R, G, B]"
        },
        "enhancement_options": {
            "enhance_images": "Apply image enhancement techniques",
            "enhance_brightness": "Brightness factor (1.0 = no change)",
            "enhance_contrast": "Contrast factor (1.0 = no change)",
            "enhance_sharpness": "Sharpness factor (1.0 = no change)",
            "enhance_color": "Color saturation factor (1.0 = no change)"
        },
        "noise_reduction_options": {
            "noise_reduction": "Apply noise reduction/smoothing",
            "noise_method": "Method: 'gaussian', 'median', 'bilateral'",
            "kernel_size": "Size of the filtering kernel"
        },
        "morphological_options": {
            "morphological_ops": "Apply morphological operations",
            "morph_operation": "Operation: 'opening', 'closing', 'gradient', 'tophat', 'blackhat'",
            "morph_kernel_size": "Size of morphological kernel",
            "morph_iterations": "Number of iterations to apply"
        },
        "binarization_options": {
            "binarization": "Convert images to binary (black and white)",
            "threshold_method": "Method: 'otsu', 'adaptive', 'binary', 'truncate'",
            "threshold_value": "Threshold value for binary methods"
        },
        "roi_options": {
            "roi_extraction": "Extract region of interest",
            "roi_method": "Method: 'contour', 'threshold', 'watershed'",
            "roi_threshold": "Threshold value for ROI extraction"
        },
        "augmentation_options": {
            "augmentation": "Apply data augmentation",
            "augmentation_factor": "Multiplier for number of augmented images",
            "horizontal_flip": "Enable horizontal flipping",
            "vertical_flip": "Enable vertical flipping",
            "rotation_range": "Maximum rotation angle in degrees",
            "zoom_range": "Zoom range as fraction",
            "brightness_range": "Brightness variation range",
            "contrast_range": "Contrast variation range",
            "blur_limit": "Maximum blur kernel size",
            "random_crop": "Enable random cropping",
            "elastic_transform": "Enable elastic deformation",
            "noise_augmentation": "Add random noise"
        }
    }


@app.post("/api/feature-engineering")
def feature_engineering(request: FeatureEngineeringRequest):
    logger.info(
        f"Starting feature engineering with parameters: "
        f"use_feature_tools={request.use_feature_tools}, "
        f"use_shap={request.use_shap}, "
        f"n_features={request.n_features}"
    )
    check_modules_available()

    try:
        result = run_feature_engineering(
            config_path="intel.yaml",
            use_feature_tools=request.use_feature_tools,
            use_shap=request.use_shap,
            n_features=request.n_features
        )
        logger.info("Feature engineering completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")


@app.get("/api/available-models")
def get_model_list():
    logger.info("Fetching available models")
    check_modules_available()
    try:
        builder = EnhancedModelBuilder()
        models = builder.get_available_models()
        logger.info(f"Retrieved {len(models)} available models")
        return models
    except Exception as e:
        logger.error(f"Error fetching available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")


@app.post("/api/build-single-model")
def build_single_model(request: ModelBuildRequest):
    logger.info(f"Starting single model building with model: {request.model_name}")
    check_modules_available()

    try:
        result = build_single_model_api_patched(
            model_name=request.model_name,
            custom_params=request.custom_params
        )
        logger.info("Single model building completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"build_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during single model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Single model building failed: {str(e)}")


@app.post("/api/build-parallel-models")
def build_parallel_models(request: ParallelModelRequest):
    logger.info(f"Starting parallel model building with {len(request.models)} models")
    check_modules_available()
    try:
        result = build_parallel_models_api_patched(
            models_config=request.models,
            max_workers=request.max_workers
        )
        logger.info("Parallel model building completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during parallel model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parallel model building failed: {str(e)}")


@app.post("/api/build-ensemble-model")
def build_ensemble_model(request: EnsembleModelRequest):
    logger.info(f"Starting ensemble model building: {request.ensemble_category} - {request.ensemble_type}")
    check_modules_available()

    try:
        # Prepare ensemble configuration based on category and type
        ensemble_config = {
            'type': request.ensemble_type,
            **request.config
        }

        # Handle different ensemble categories
        if request.ensemble_category == 'multi_stage':
            ensemble_config['type'] = 'multi_stage_stacking'
        elif request.ensemble_category == 'hybrid':
            # Keep the original ensemble_type for hybrid ensembles
            pass

        # Add debug logging
        logger.info(f"Ensemble config: {ensemble_config}")

        result = build_ensemble_model_api(ensemble_config=ensemble_config)
        logger.info("Ensemble model building completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"build_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during ensemble model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ensemble model building failed: {str(e)}")


# NEW: Staged Ensemble API Endpoint
@app.post("/api/build-staged-ensemble")
def build_staged_ensemble(request: StagedEnsembleRequest):
    logger.info(f"Starting staged ensemble model building with {len(request.architecture)} stages")
    check_modules_available()

    try:
        staged_config = {
            'type': 'staged_ensemble',
            'architecture': request.architecture,
            'final_meta_model': request.final_meta_model,
            'cv': request.cv,
            'stage_method': request.stage_method,
            'combine_method': request.combine_method,
            'passthrough_original': request.passthrough_original
        }

        logger.info(f"Staged ensemble config: {staged_config}")

        result = build_staged_ensemble_api(staged_config=staged_config)
        logger.info("Staged ensemble model building completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"build_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during staged ensemble model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Staged ensemble model building failed: {str(e)}")


@app.post("/api/debug-ensemble-config")
def debug_ensemble_config(config: Dict[str, Any] = Body(...)):
    """Debug endpoint to validate ensemble configuration"""
    try:
        logger.info(f"Debug ensemble config received: {config}")

        # Validate configuration
        required_fields = ['type']
        for field in required_fields:
            if field not in config:
                return {"error": f"Missing required field: {field}"}

        ensemble_type = config.get('type')

        if ensemble_type == 'voting':
            if 'models' not in config:
                return {"error": "Voting ensemble requires 'models' field"}
            if len(config['models']) < 2:
                return {"error": "Voting ensemble requires at least 2 models"}
        elif ensemble_type == 'stacking':
            if 'base_models' not in config:
                return {"error": "Stacking ensemble requires 'base_models' field"}
            if 'meta_model' not in config:
                return {"error": "Stacking ensemble requires 'meta_model' field"}
        elif ensemble_type == 'bagging':
            if 'base_model' not in config:
                return {"error": "Bagging ensemble requires 'base_model' field"}
        elif ensemble_type == 'staged_ensemble':
            if 'architecture' not in config:
                return {"error": "Staged ensemble requires 'architecture' field"}
            if not isinstance(config['architecture'], list) or len(config['architecture']) < 2:
                return {"error": "Staged ensemble requires at least 2 stages in architecture"}

        return {
            "status": "valid",
            "config": config,
            "message": f"Configuration for {ensemble_type} ensemble is valid"
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {"error": str(e)}


@app.post("/api/build-simple-ensemble")
def build_simple_ensemble(request: SimpleEnsembleRequest):
    """Build simple ensemble models (voting, stacking, bagging)"""
    logger.info(f"Starting simple ensemble model building: {request.ensemble_type}")
    check_modules_available()

    try:
        ensemble_config = {
            'type': request.ensemble_type,
            **request.config
        }

        result = build_ensemble_model_api(ensemble_config=ensemble_config)
        logger.info("Simple ensemble model building completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"build_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during simple ensemble model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple ensemble model building failed: {str(e)}")


@app.post("/api/build-multi-stage-stacking")
def build_multi_stage_stacking(request: MultiStageStackingRequest):
    """Build multi-stage stacking ensemble models"""
    logger.info(f"Starting multi-stage stacking model building with {len(request.stages)} stages")
    check_modules_available()

    try:
        ensemble_config = {
            'type': 'multi_stage_stacking',
            'stages': request.stages,
            'final_estimator': request.final_estimator,
            'cv': request.cv,
            'passthrough': request.passthrough
        }

        result = build_ensemble_model_api(ensemble_config=ensemble_config)
        logger.info("Multi-stage stacking model building completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"build_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during multi-stage stacking model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-stage stacking model building failed: {str(e)}")


@app.post("/api/build-hybrid-ensemble")
def build_hybrid_ensemble(request: HybridEnsembleRequest):
    """Build hybrid ensemble models (voting+stacking, bagging+stacking)"""
    logger.info(f"Starting hybrid ensemble model building: {request.ensemble_type}")
    check_modules_available()

    try:
        ensemble_config = {
            'type': request.ensemble_type,
            **request.config
        }

        result = build_ensemble_model_api(ensemble_config=ensemble_config)
        logger.info("Hybrid ensemble model building completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"build_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during hybrid ensemble model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid ensemble model building failed: {str(e)}")


@app.post("/api/select-best-model")
def select_best_model(request: ModelSelectionRequest):
    logger.info(f"Selecting best model based on {request.selection_metric}")
    check_modules_available()

    try:
        result = select_best_model_api_patched(
            parallel_results=request.parallel_results,
            selection_metric=request.selection_metric
        )
        logger.info("Model selection completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"selection_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during model selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model selection failed: {str(e)}")


# Keep the legacy endpoint for backwards compatibility
@app.post("/api/build-model")
def build_model(request: ModelBuildRequest):
    logger.info(f"Starting model building with model: {request.model_name}")
    return build_single_model(request)


@app.post("/api/optimize")
def optimize(request: OptimizationRequest):
    if not request.optimize:
        logger.info("Optimization skipped by user request")
        return {"message": "Optimization skipped."}

    logger.info(
        f"Starting model optimization with method: {request.method}, n_trials: {request.n_trials, metric: {request.metric}}")
    check_modules_available()

    try:
        result = optimize_model(
            optimize=request.optimize,
            method=request.method,
            n_trials=request.n_trials,
            metric=request.metric,
            config_overrides=None
        )
        logger.info("Model optimization completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Post-optimization evaluation completed successfully")

        return {"optimization_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model optimization failed: {str(e)}")


@app.get("/api/download-model")
def download_model():
    logger.info("Model download requested")
    check_modules_available()
    try:
        intel = load_yaml("intel.yaml")
        dataset_name = intel.get("dataset_name", "unnamed")
        model_dir = Path(f"model/model_{dataset_name}")

        # Use absolute paths for checking existence
        optimized_model_path = model_dir / "optimized_model.pkl"
        standard_model_path = model_dir / "model.pkl"

        if optimized_model_path.exists():
            model_path = optimized_model_path
            filename = f"optimized_model_{dataset_name}.pkl"
        elif standard_model_path.exists():
            model_path = standard_model_path
            filename = f"model_{dataset_name}.pkl"
        else:
            # Get absolute path from intel.yaml
            model_path = Path(intel.get("model_path", "")).resolve()
            filename = model_path.name if model_path else "model.pkl"

        if not model_path or not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            raise HTTPException(status_code=404, detail="Model file not found")

        return FileResponse(
            path=str(model_path),
            filename=filename,
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")


@app.get("/api/download-pipeline")
def download_pipeline():
    logger.info("Pipeline download requested")
    check_modules_available()
    try:
        intel = load_yaml("intel.yaml")
        dataset_name = intel.get("dataset_name", "unnamed")

        processor_path = Path(f"model/pipelines/preprocessing_{dataset_name}/processor.pkl")
        if processor_path.exists():
            logger.info(f"Serving combined pipeline: combined_pipeline_{dataset_name}.pkl")
            return FileResponse(
                path=str(processor_path),
                filename=f"combined_pipeline_{dataset_name}.pkl",
                media_type="application/octet-stream"
            )

        pipeline_dir = Path(f"model/pipelines/preprocessing_{dataset_name}")
        preprocessing_path = pipeline_dir / "processor.pkl"
        if preprocessing_path.exists():
            logger.info(f"Serving preprocessing pipeline: preprocessing_pipeline_{dataset_name}.pkl")
            return FileResponse(
                path=str(preprocessing_path),
                filename=f"preprocessing_pipeline_{dataset_name}.pkl",
                media_type="application/octet-stream"
            )

        pipeline_path = intel.get("preprocessing_pipeline_path")
        if not pipeline_path or not os.path.exists(pipeline_path):
            logger.error("Pipeline file not found")
            raise HTTPException(status_code=404, detail="Pipeline file not found")

        logger.info(f"Serving fallback pipeline: {os.path.basename(pipeline_path)}")
        return FileResponse(
            path=pipeline_path,
            filename=os.path.basename(pipeline_path),
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading pipeline: {str(e)}")


@app.get("/api/download-feature-pipeline")
def download_feature_pipeline():
    logger.info("Feature pipeline download requested")
    check_modules_available()
    try:
        intel = load_yaml("intel.yaml")
        dataset_name = intel.get("dataset_name", "unnamed")

        feature_pipeline_path = Path(f"model/pipelines/preprocessing_{dataset_name}/transformation.pkl")

        if not feature_pipeline_path.exists():
            logger.error("Feature engineering pipeline not found")
            raise HTTPException(status_code=404, detail="Feature engineering pipeline not found")

        logger.info(f"Serving feature pipeline: feature_pipeline_{dataset_name}.pkl")
        return FileResponse(
            path=str(feature_pipeline_path),
            filename=f"feature_pipeline_{dataset_name}.pkl",
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading feature pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading feature pipeline: {str(e)}")


@app.get("/api/generate-report")
def generate_report():
    logger.info("Report generation requested")
    check_modules_available()
    try:
        from semiauto_classification.visualization.projectflow_report import ClassificationProjectFlowReport
        report_generator = ClassificationProjectFlowReport("intel.yaml")
        report_generator.generate_report()
        intel = load_yaml("intel.yaml")
        dataset_name = intel.get("dataset_name", "unnamed")
        report_path = f"reports/pdf/projectflow_report_{dataset_name}.pdf"

        if not os.path.exists(report_path):
            logger.error("Report generation failed - file not found")
            raise HTTPException(status_code=500, detail="Report generation failed")

        logger.info(f"Serving generated report: {os.path.basename(report_path)}")
        return FileResponse(
            path=report_path,
            filename=os.path.basename(report_path),
            media_type="application/pdf"
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.post("/api/feature-selection")
async def run_feature_selection(
        algorithm: str = Body(...),
        n_features: Optional[int] = Body(None),
        max_iter: int = Body(50),
        population_size: int = Body(20)
):
    check_modules_available()
    try:
        # Run feature selection pipeline
        feature_selection_pipeline(
            intel_path="intel.yaml",
            algorithm=algorithm,
            n_features=n_features,
            max_iter=max_iter,
            population_size=population_size
        )

        # Load results to return
        intel = load_yaml("intel.yaml")
        config = intel.get('feature_selection_config', {})

        return {
            "status": "success",
            "message": "Feature selection completed",
            "selected_features": config.get('selected_features', []),
            "cv_accuracy": 0.95  # Placeholder - actual value would come from the pipeline
        }
    except Exception as e:
        logger.error(f"Error during feature selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature selection failed: {str(e)}")


@app.get("/api/feature-selection/algorithms")
async def get_feature_selection_algorithms():
    check_modules_available()
    try:
        selector = MetaHeuristicFeatureSelector()
        algorithms = selector.get_available_algorithms()
        return JSONResponse(content=algorithms)
    except Exception as e:
        logger.error(f"Error getting feature selection algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting feature selection algorithms: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("classification-static/css", exist_ok=True)
    os.makedirs("classification-static/js", exist_ok=True)
    os.makedirs("classification-static/images", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    logger.info("Starting FastAPI server on 127.0.0.1:8080")
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)