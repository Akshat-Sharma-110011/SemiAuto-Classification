import json
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import datetime
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend


class CustomPDF(FPDF):
    def __init__(self, bg_color, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_color = bg_color

    def header(self):
        """Draw the dark background on every page"""
        # Fill the background with specified color
        self.set_fill_color(*self.bg_color)
        self.rect(0, 0, 210, 297, style='F')  # A4 size in mm


class ClassificationProjectFlowReport:
    def __init__(self, intel_yaml_path):
        """Initialize with the path to the intel.yaml file"""
        with open(intel_yaml_path, 'r') as file:
            self.intel = yaml.safe_load(file)

        self.dataset_name = self.intel.get('dataset_name', 'unknown')
        self.output_path = f"reports/pdf/projectflow_report_{self.dataset_name}.pdf"

        # Create the PDF directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Set up color scheme
        self.bg_color = (32, 34, 22)  # Dark olive background
        self.primary_color = (242, 222, 155)  # Pale gold text
        self.highlight_color = (255, 240, 193)  # Lighter gold
        self.secondary_color = (212, 193, 114)  # Darker gold

        # Initialize PDF with custom class
        self.pdf = CustomPDF(bg_color=self.bg_color)
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.add_page()

    def add_page(self):
        """Add a new page"""
        self.pdf.add_page()

    def add_title_page(self):
        """Add a title page to the PDF"""
        self.pdf.set_font('Helvetica', 'B', 24)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 20, "SemiAuto Classification Report", 0, 1, 'C')

        self.pdf.set_font('Helvetica', 'B', 18)
        self.pdf.cell(0, 15, f"Dataset: {self.dataset_name.upper()}", 0, 1, 'C')

        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')

        # Project flow diagram
        self.pdf.ln(20)
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.cell(0, 10, "Project Flow", 0, 1, 'C')

        steps = ["Data Ingestion", "Data Preprocessing", "Feature Engineering",
                 "Model Building", "Model Evaluation", "Model Optimization",
                 "Final Evaluation"]

        self.pdf.ln(5)
        self.pdf.set_draw_color(*self.primary_color)
        self.pdf.set_line_width(0.5)

        y_pos = self.pdf.get_y()
        x_start = 30
        x_end = 180
        y_arrow = y_pos + 20

        self.pdf.line(x_start, y_arrow, x_end, y_arrow)
        self.pdf.line(x_end, y_arrow, x_end - 5, y_arrow - 3)
        self.pdf.line(x_end, y_arrow, x_end - 5, y_arrow + 3)

        step_width = (x_end - x_start) / (len(steps) - 1)
        self.pdf.set_font('Helvetica', '', 9)

        for i, step in enumerate(steps):
            x_pos = x_start + (i * step_width)
            self.pdf.set_fill_color(*self.secondary_color)
            self.pdf.ellipse(x_pos - 3, y_arrow - 3, 6, 6, 'F')
            self.pdf.set_xy(x_pos - 15, y_arrow + 5)
            self.pdf.cell(30, 10, step, 0, 0, 'C')

        self.pdf.ln(40)

        # Table of Contents
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.cell(0, 10, "Table of Contents", 0, 1, 'L')

        toc_items = [
            "1. Data Ingestion",
            "2. Data Preprocessing",
            "3. Feature Engineering",
            "4. Model Building",
            "5. Model Evaluation",
            "6. Model Optimization (if performed)",
            "7. Final Evaluation Results"
        ]

        self.pdf.set_font('Helvetica', '', 12)
        for item in toc_items:
            self.pdf.cell(0, 8, item, 0, 1, 'L')

        self.add_page()

    def add_section_header(self, title, description=""):
        """Add a section header"""
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 15, title, 0, 1, 'L')

        if description:
            self.pdf.set_font('Helvetica', '', 11)
            self.pdf.multi_cell(0, 6, description)
            self.pdf.ln(5)

    def add_subsection_header(self, title):
        """Add a subsection header"""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.set_text_color(*self.secondary_color)
        self.pdf.cell(0, 10, title, 0, 1, 'L')

    def add_data_ingestion_section(self):
        """Add data ingestion section"""
        self.add_section_header("1. Data Ingestion",
                                "Initial dataset analysis and characteristics.")

        self.add_subsection_header("Dataset Overview")
        feature_store_path = self.intel.get('feature_store_path')
        if feature_store_path and os.path.exists(feature_store_path):
            with open(feature_store_path, 'r') as file:
                feature_store = yaml.safe_load(file)

            self.pdf.set_font('Helvetica', 'B', 12)
            self.pdf.cell(0, 8, f"Dataset: {self.dataset_name}", 0, 1, 'L')
            train_rows = feature_store.get('train_rows', 'N/A')
            test_rows = feature_store.get('test_rows', 'N/A')
            self.pdf.cell(0, 8, f"Train samples: {train_rows}, Test samples: {test_rows}", 0, 1, 'L')
            self.pdf.cell(0, 8, f"Target column: {feature_store.get('target_col', 'N/A')}", 0, 1, 'L')
            self.pdf.ln(5)

            self.add_subsection_header("Column Types")
            self.pdf.set_font('Helvetica', '', 10)

            column_info = [
                {"title": "Numerical Columns", "data": feature_store.get('numerical_cols', [])},
                {"title": "Categorical Columns", "data": feature_store.get('categorical_cols', [])},
                {"title": "Columns with Nulls", "data": feature_store.get('contains_null', [])},
                {"title": "Columns with Outliers", "data": feature_store.get('contains_outliers', [])}
            ]

            for info in column_info:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, info["title"] + ":", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.multi_cell(0, 6, ", ".join(info["data"]))
                self.pdf.ln(3)

            # Add distribution plots
            plots_dir = self.intel.get('plots_dir')
            if plots_dir and os.path.exists(plots_dir):
                self.add_page()
                self.add_subsection_header("Feature Distributions")

                dist_plots = [f for f in os.listdir(plots_dir) if f.startswith('distribution_')]

                for i in range(0, len(dist_plots), 2):
                    y_pos = self.pdf.get_y()
                    plot_path = os.path.join(plots_dir, dist_plots[i])
                    feature_name = dist_plots[i].replace('distribution_', '').replace('.png', '')
                    self.pdf.set_font('Helvetica', 'B', 10)
                    self.pdf.cell(0, 6, feature_name, 0, 1, 'C')
                    self.pdf.image(plot_path, x=25, y=None, w=75)

                    if i + 1 < len(dist_plots):
                        self.pdf.set_xy(110, y_pos)
                        plot_path = os.path.join(plots_dir, dist_plots[i + 1])
                        feature_name = dist_plots[i + 1].replace('distribution_', '').replace('.png', '')
                        self.pdf.cell(0, 6, feature_name, 0, 1, 'C')
                        self.pdf.image(plot_path, x=110, y=None, w=75)

                    self.pdf.ln(5)
                    if self.pdf.get_y() > 240:
                        self.add_page()

                # Add class distribution plot
                class_plot = os.path.join(plots_dir, 'class_distribution.png')
                if os.path.exists(class_plot):
                    self.add_page()
                    self.add_subsection_header("Class Distribution")
                    self.pdf.image(class_plot, x=25, y=None, w=160)

        self.add_page()

    def add_data_preprocessing_section(self):
        """Add data preprocessing section"""
        self.add_section_header("2. Data Preprocessing",
                                "Data cleaning and preparation steps.")

        if 'preprocessing_config' in self.intel:
            preproc_config = self.intel['preprocessing_config']
            self.add_subsection_header("Preprocessing Configuration")

            # Handle categorical encoding
            if 'categorical_encoding' in preproc_config:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, "Categorical Encoding:", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.cell(0, 6, f"Method: {preproc_config['categorical_encoding'].get('method', 'N/A')}", 0, 1, 'L')
                self.pdf.cell(0, 6, f"Drop first: {preproc_config['categorical_encoding'].get('drop_first', False)}", 0,
                              1, 'L')
                self.pdf.ln(3)

            # Handle class imbalance
            if 'class_imbalance' in preproc_config:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, "Class Imbalance Handling:", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.cell(0, 6, f"Method: {preproc_config['class_imbalance'].get('method', 'N/A')}", 0, 1, 'L')
                self.pdf.ln(3)

        self.add_page()

    def add_feature_engineering_section(self):
        """Add feature engineering section"""
        self.add_section_header("3. Feature Engineering",
                                "Feature selection and engineering steps.")

        if 'feature_engineering_config' in self.intel:
            feat_config = self.intel['feature_engineering_config']
            self.add_subsection_header("Feature Engineering Config")

            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.cell(0, 8, "Feature Selection Method:", 0, 1, 'L')
            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.cell(0, 6, f"{'SHAP' if feat_config.get('use_shap_selection', False) else 'None'}", 0, 1, 'L')

            # Add feature importance plot
            plots_dir = self.intel.get('plots_dir')
            if plots_dir and os.path.exists(plots_dir):
                feat_plot = os.path.join(plots_dir, 'feature_importance.png')
                if os.path.exists(feat_plot):
                    self.add_page()
                    self.add_subsection_header("Feature Importance")
                    self.pdf.image(feat_plot, x=25, y=None, w=160)

        self.add_page()

    def add_model_building_section(self):
        """Add model building section"""
        self.add_section_header("4. Model Building",
                                "Model training configuration.")

        if 'model_name' in self.intel:
            self.add_subsection_header("Model Selection")
            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.cell(0, 8, "Selected Model:", 0, 1, 'L')
            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.cell(0, 6, self.intel['model_name'], 0, 1, 'L')

        self.add_page()

    def add_model_evaluation_section(self):
        """Add model evaluation section"""
        self.add_section_header("5. Model Evaluation",
                                "Model performance metrics.")

        if 'performance_metrics_path' in self.intel:
            metrics_path = self.intel['performance_metrics_path']
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as file:
                    metrics = yaml.safe_load(file)

                self.add_subsection_header("Classification Metrics")

                metric_data = [
                    ("Accuracy", metrics.get('accuracy', 'N/A')),
                    ("Precision", metrics.get('precision', 'N/A')),
                    ("Recall", metrics.get('recall', 'N/A')),
                    ("F1 Score", metrics.get('f1_score', 'N/A')),
                    ("ROC AUC", metrics.get('roc_auc', 'N/A'))
                ]

                # Create metrics table
                self.pdf.set_font('Helvetica', 'B', 10)
                self.pdf.set_fill_color(*self.highlight_color)
                self.pdf.set_text_color(*self.bg_color)
                self.pdf.cell(90, 8, "Metric", 1, 0, 'C', True)
                self.pdf.cell(90, 8, "Value", 1, 1, 'C', True)
                self.pdf.set_text_color(*self.primary_color)

                self.pdf.set_font('Helvetica', '', 10)
                for metric, value in metric_data:
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    self.pdf.cell(90, 8, metric, 1, 0, 'L')
                    self.pdf.cell(90, 8, str(value), 1, 1, 'C')

                # Add confusion matrix
                plots_dir = self.intel.get('plots_dir')
                if plots_dir and os.path.exists(plots_dir):
                    cm_plot = os.path.join(plots_dir, 'confusion_matrix.png')
                    if os.path.exists(cm_plot):
                        self.add_page()
                        self.add_subsection_header("Confusion Matrix")
                        self.pdf.image(cm_plot, x=25, y=None, w=160)

        self.add_page()

    def add_model_optimization_section(self):
        """Add model optimization section"""
        if 'optimized_model_path' not in self.intel:
            self.add_section_header("6. Model Optimization",
                                    "This step was skipped.")
            self.add_page()
            return

        self.add_section_header("6. Model Optimization",
                                "Hyperparameter tuning results.")

        if 'best_params_path' in self.intel:
            params_path = self.intel['best_params_path']
            if os.path.exists(params_path):
                try:
                    # Check if file is not empty
                    if os.path.getsize(params_path) == 0:
                        raise ValueError("Hyperparameters file is empty")

                    with open(params_path, 'r') as file:
                        params = json.load(file)

                    self.add_subsection_header("Optimized Parameters")

                    self.pdf.set_font('Helvetica', 'B', 10)
                    self.pdf.set_fill_color(*self.highlight_color)
                    self.pdf.set_text_color(*self.bg_color)
                    self.pdf.cell(90, 8, "Parameter", 1, 0, 'C', True)
                    self.pdf.cell(90, 8, "Value", 1, 1, 'C', True)
                    self.pdf.set_text_color(*self.primary_color)

                    self.pdf.set_font('Helvetica', '', 10)
                    for param, value in params.items():
                        self.pdf.cell(90, 8, param, 1, 0, 'L')
                        self.pdf.cell(90, 8, str(value), 1, 1, 'C')

                except json.JSONDecodeError:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, "Error: Invalid JSON format in hyperparameters file", 0, 1, 'L')
                except ValueError as ve:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, str(ve), 0, 1, 'L')
                except Exception as e:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, f"Error loading hyperparameters: {str(e)}", 0, 1, 'L')

        self.add_page()

    def add_final_evaluation_section(self):
        """Add final evaluation section"""
        self.add_section_header("7. Final Evaluation Results",
                                "Optimized model performance.")

        if 'optimized_performance_metrics_path' not in self.intel:
            self.pdf.cell(0, 8, "No optimization performed", 0, 1, 'L')
            return

        metrics_path = self.intel['optimized_performance_metrics_path']
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as file:
                metrics = yaml.safe_load(file)

            self.add_subsection_header("Optimized Metrics")

            metric_data = [
                ("Accuracy", metrics.get('accuracy', 'N/A')),
                ("Precision", metrics.get('precision', 'N/A')),
                ("Recall", metrics.get('recall', 'N/A')),
                ("F1 Score", metrics.get('f1_score', 'N/A')),
                ("ROC AUC", metrics.get('roc_auc', 'N/A'))
            ]

            # Create comparison table
            self.pdf.set_font('Helvetica', 'B', 10)
            self.pdf.set_fill_color(*self.highlight_color)
            self.pdf.set_text_color(*self.bg_color)
            self.pdf.cell(50, 8, "Metric", 1, 0, 'C', True)
            self.pdf.cell(50, 8, "Original", 1, 0, 'C', True)
            self.pdf.cell(50, 8, "Optimized", 1, 0, 'C', True)
            self.pdf.cell(30, 8, "Change", 1, 1, 'C', True)
            self.pdf.set_text_color(*self.primary_color)

            # Add comparison data
            orig_metrics = self._load_original_metrics()
            self.pdf.set_font('Helvetica', '', 10)
            for metric, opt_val in metric_data:
                orig_val = orig_metrics.get(metric.lower().replace(' ', '_'), 'N/A')
                if isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)):
                    change = opt_val - orig_val
                    change_str = f"{change:.4f}"
                    color = (46, 204, 113) if change > 0 else (231, 76, 60)
                else:
                    change_str = "N/A"
                    color = self.primary_color

                self.pdf.cell(50, 8, metric, 1, 0, 'L')
                self.pdf.cell(50, 8, str(orig_val), 1, 0, 'C')
                self.pdf.cell(50, 8, str(opt_val), 1, 0, 'C')
                self.pdf.set_text_color(*color)
                self.pdf.cell(30, 8, change_str, 1, 1, 'C')
                self.pdf.set_text_color(*self.primary_color)

    def _load_original_metrics(self):
        """Helper to load original metrics"""
        if 'performance_metrics_path' in self.intel:
            metrics_path = self.intel['performance_metrics_path']
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as file:
                    return yaml.safe_load(file)
        return {}

    def generate_report(self):
        """Generate complete report"""
        self.add_title_page()
        self.add_data_ingestion_section()
        self.add_data_preprocessing_section()
        self.add_feature_engineering_section()
        self.add_model_building_section()
        self.add_model_evaluation_section()
        self.add_model_optimization_section()
        self.add_final_evaluation_section()

        self.pdf.output(self.output_path)
        print(f"Report generated: {self.output_path}")


if __name__ == "__main__":
    report = ClassificationProjectFlowReport("intel.yaml")
    report.generate_report()