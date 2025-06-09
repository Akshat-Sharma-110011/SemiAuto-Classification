from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
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
from semiauto_classification.logger import configure_logger, get_logger

configure_logger()  # Configure logging once at startup

# Get logger for this module
logger = get_logger("FastAPI-App")

# Import internal modules after logging is configured
from semiauto_classification.data.data_ingestion import create_data_ingestion
from semiauto_classification.utils import (load_yaml, update_intel_yaml)
from semiauto_classification.data.data_cleaning import main as data_cleaning_main
from semiauto_classification.features.feature_engineering import run_feature_engineering
from semiauto_classification.model.model_building import ModelBuilder
from semiauto_classification.model.model_evaluation import run_evaluation, get_evaluation_summary
from semiauto_classification.model.model_optimization import optimize_model

# Log application startup
logger.info("Starting SemiAuto Classification FastAPI application")

app = FastAPI(title="SemiAuto Classification", version="1.0")

# Set up static files and templates
app.mount("/classification-static", StaticFiles(directory="static"), name="classification-static")
templates = Jinja2Templates(directory="templates")
templates.env.filters["zip"] = zip
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

logger.info("FastAPI application and middleware configured")


class PreprocessingConfig(BaseModel):
    missing_values: Optional[str] = None
    handle_duplicates: bool = True
    outliers: Optional[str] = None
    skewedness: Optional[str] = None
    scaling: Optional[str] = None
    encoding: Optional[str] = None
    drop_first: Optional[bool] = False


class FeatureEngineeringRequest(BaseModel):
    use_feature_tools: bool = False
    use_shap: bool = False
    n_features: int = 20


class ModelBuildRequest(BaseModel):
    model_name: str
    custom_params: Optional[Dict[str, Any]] = None


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
    return templates.TemplateResponse("data_upload.html", {"request": request})


# Route to preprocessing page
@app.get("/preprocessing", response_class=HTMLResponse)
async def preprocessing_page(request: Request):
    logger.info("Serving preprocessing page")
    try:
        intel = load_yaml("intel.yaml")
        feature_store = {}
        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])

        logger.info(f"Loaded preprocessing page data for dataset: {intel.get('dataset_name', 'Unknown')}")
        return templates.TemplateResponse("preprocessing.html", {
            "request": request,
            "dataset_name": intel.get('dataset_name', ''),
            "target_column": intel.get('target_column', ''),
            "numerical_cols": feature_store.get('numerical_cols', []),
            "categorical_cols": feature_store.get('categorical_cols', []),
            "nulls": feature_store.get('contains_null', []),
            "outliers": feature_store.get('contains_outliers', []),
            "skewed": feature_store.get('skewed_cols', [])
        })
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
        intel = load_yaml("intel.yaml")
        if not intel.get('train_preprocessed_path'):
            logger.warning("Preprocessing not completed, redirecting to preprocessing page")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please complete preprocessing first",
                "redirect_url": "/preprocessing"
            })
        return templates.TemplateResponse("feature_engineering.html", {"request": request})
    except Exception as e:
        logger.error(f"Error loading feature engineering page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please upload data and complete preprocessing first",
            "redirect_url": "/data-upload"
        })


# Route to model building page
@app.get("/model-building", response_class=HTMLResponse)
async def model_building_page(request: Request):
    logger.info("Serving model building page")
    try:
        intel = load_yaml("intel.yaml")
        if not intel.get('train_transformed_path'):
            logger.warning("Feature engineering not completed, redirecting to feature engineering page")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please complete feature engineering first",
                "redirect_url": "/feature-engineering"
            })

        builder = ModelBuilder()
        available_models = builder.get_available_models()
        logger.info(f"Loaded {len(available_models)} available model")

        return templates.TemplateResponse("model_building.html", {
            "request": request,
            "available_models": available_models
        })
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
        intel = load_yaml("intel.yaml")
        if not intel.get('model_path'):
            logger.warning("Model not built, redirecting to model building page")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please build a model first",
                "redirect_url": "/model-building"
            })

        return templates.TemplateResponse("optimization.html", {"request": request})
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

        return templates.TemplateResponse("results.html", {
            "request": request,
            "metrics": metrics,
            "dataset_name": intel.get('dataset_name', 'unnamed')
        })
    except Exception as e:
        logger.error(f"Error loading results page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# API endpoints
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), target_column: str = Form(...)):
    logger.info(f"Received file upload: {file.filename}, target column: {target_column}")

    if not file.filename.endswith(".csv"):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

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
            ingestion.run_ingestion_pipeline(f, file.filename, target_column)

        os.unlink(path)
        ingestion.save_intel_yaml()
        logger.info("Data ingestion completed successfully")

        try:
            data_cleaning_main()
            logger.info("Data cleaning completed successfully")
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")

        return {
            "message": "Data ingestion and cleaning completed",
            "columns": columns
        }
    except Exception as e:
        logger.error(f"Error during data upload: {str(e)}")
        if os.path.exists(path):
            os.unlink(path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/preprocess")
def preprocess_data(config: PreprocessingConfig):
    logger.info("Starting data preprocessing")

    from semiauto_classification.data.data_preprocessing import (
        PreprocessingPipeline, PreprocessingParameters,
        check_for_duplicates, get_numerical_columns,
        get_categorical_columns, recommend_skewness_transformer,
        load_yaml, update_intel_yaml
    )

    try:
        intel = load_yaml("intel.yaml")
        train_df = pd.read_csv(intel['cleaned_train_path'])
        test_df = pd.read_csv(intel['cleaned_test_path']) if 'cleaned_test_path' in intel else pd.DataFrame()

        feature_store = load_yaml(intel['feature_store_path'])
        logger.info(f"Loaded data for preprocessing: {len(train_df)} train rows, {len(test_df)} test rows")

        numerical = feature_store.get('numerical_cols', [])
        categorical = feature_store.get('categorical_cols', [])
        nulls = [col for col in feature_store.get('contains_null', []) if col != intel['target_column']]
        outliers = [col for col in feature_store.get('contains_outliers', []) if col != intel['target_column']]
        skewed = [col for col in feature_store.get('skewed_cols', []) if col != intel['target_column']]

        pipeline = PreprocessingPipeline({
            'dataset_name': intel['dataset_name'],
            'target_col': intel['target_column'],
            'feature_store': feature_store
        }, PreprocessingParameters())

        if config.missing_values and nulls:
            logger.info(f"Handling missing values with method: {config.missing_values}")
            pipeline.handle_missing_values(config.missing_values, nulls)
        if config.outliers and outliers:
            logger.info(f"Handling outliers with method: {config.outliers}")
            pipeline.handle_outliers(config.outliers, outliers)
        if config.skewedness and skewed:
            logger.info(f"Handling skewed data with method: {config.skewedness}")
            pipeline.handle_skewed_data(config.skewedness, skewed)
        if config.scaling and numerical:
            logger.info(f"Scaling numerical features with method: {config.scaling}")
            pipeline.scale_numerical_features(config.scaling, numerical)
        if config.encoding and categorical:
            logger.info(f"Encoding categorical features with method: {config.encoding}")
            pipeline.encode_categorical_features(config.encoding, categorical, config.drop_first)

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
        return {"message": "Preprocessing completed"}

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@app.post("/api/feature-engineering")
def feature_engineering(request: FeatureEngineeringRequest):
    logger.info(
        f"Starting feature engineering with parameters: use_feature_tools={request.use_feature_tools}, use_shap={request.use_shap}, n_features={request.n_features}")

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
    logger.info("Fetching available model")
    try:
        builder = ModelBuilder()
        models = builder.get_available_models()
        logger.info(f"Retrieved {len(models)} available model")
        return models
    except Exception as e:
        logger.error(f"Error fetching available model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available model: {str(e)}")


@app.post("/api/build-model")
def build_model(request: ModelBuildRequest):
    logger.info(f"Starting model building with model: {request.model_name}")

    try:
        builder = ModelBuilder()
        result = builder.process_model_request(
            model_name=request.model_name,
            custom_params=request.custom_params
        )
        logger.info("Model building completed successfully")

        evaluation = run_evaluation("intel.yaml")
        logger.info("Model evaluation completed successfully")

        return {"build_result": result, "evaluation_result": evaluation}
    except Exception as e:
        logger.error(f"Error during model building: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model building failed: {str(e)}")


@app.post("/api/optimize")
def optimize(request: OptimizationRequest):
    if not request.optimize:
        logger.info("Optimization skipped by user request")
        return {"message": "Optimization skipped."}

    logger.info(
        f"Starting model optimization with method: {request.method}, n_trials: {request.n_trials}, metric: {request.metric}")

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
    try:
        intel = load_yaml("intel.yaml")
        dataset_name = intel.get("dataset_name", "unnamed")
        model_dir = Path(f"model/model_classification_{dataset_name}")

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


if __name__ == "__main__":
    os.makedirs("classification-static/css", exist_ok=True)
    os.makedirs("classification-static/js", exist_ok=True)
    os.makedirs("classification-static/images", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    logger.info("Starting FastAPI server on 127.0.0.1:8020")
    uvicorn.run("app:app", host="127.0.0.1", port=8020, reload=True)