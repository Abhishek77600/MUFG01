import json
import os

def display_all_metrics():
    """Display all model metrics from metadata in a clear, professional format"""

    # Load metadata
    artifacts_dir = 'artifacts'
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("❌ Metadata file not found. Please run training first.")
        return

    evaluations = metadata.get('evaluations', {})
    chosen_model = metadata.get('chosen_model', 'Unknown')

    print("=" * 100)
    print("HEART DISEASE PREDICTION MODEL METRICS DASHBOARD")
    print("=" * 100)
    print(f"BEST MODEL: {chosen_model.replace('_', ' ').upper()}")
    print()

    # Display all model metrics
    print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
    print("-" * 100)

    # Header
    print("<25")
    print("-" * 100)

    # Sort models by ROC-AUC
    sorted_models = sorted(evaluations.items(), key=lambda x: x[1]['test_eval']['roc_auc'], reverse=True)

    for model_name, eval_data in sorted_models:
        test_eval = eval_data['test_eval']
        report = test_eval['classification_report']

        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        roc_auc = test_eval['roc_auc']

        # Highlight best model
        marker = " *** BEST ***" if model_name == chosen_model else ""
        model_display = model_name.replace('_', ' ').title()

        print("<25")

    print()

    # Display detailed best model metrics
    print("DETAILED BEST MODEL PERFORMANCE")
    print("-" * 50)

    best_eval = evaluations[chosen_model]['test_eval']
    report = best_eval['classification_report']

    print(f"Model: {chosen_model.replace('_', ' ').title()}")
    print(f"ROC-AUC: {best_eval['roc_auc']:.4f}")
    print()

    print("CLASSIFICATION REPORT:")
    print("Class 0 (No Heart Disease):")
    print(".4f")
    print(".4f")
    print(".4f")
    print()

    print("Class 1 (Heart Disease):")
    print(".4f")
    print(".4f")
    print(".4f")
    print()

    print("Overall Metrics:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print()

    # Display best parameters
    print("BEST HYPERPARAMETERS:")
    print("-" * 30)
    best_params = evaluations[chosen_model].get('best_params', {})
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print()

    # Display cross-validation score
    cv_score = evaluations[chosen_model].get('cv_score', 0)
    print(".4f")
    print()

    # Display available plots
    print("AVAILABLE VISUALIZATIONS")
    print("-" * 30)

    plots_dir = 'plots'
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        plot_files.sort()

        plot_categories = {
            "Model Performance": [
                "model_performance_comparison.png",
                "roc_curves_optimized.png",
                "enhanced_confusion_matrices.png",
                "model_improvement_summary.png"
            ],
            "Feature Analysis": [
                "feature_importances.png",
                "feature_importance_waterfall.png",
                "correlation_heatmap.png",
                "enhanced_correlation_heatmap.png"
            ],
            "Data Exploration": [
                "dataset_overview.png",
                "feature_analysis_dashboard.png",
                "health_metrics_dashboard.png",
                "outlier_analysis.png"
            ],
            "Distributions": [
                f for f in plot_files if f.startswith(('hist_', 'box_'))
            ]
        }

        for category, plots in plot_categories.items():
            available_plots = [p for p in plots if p in plot_files]
            if available_plots:
                print(f"{category}:")
                for plot in available_plots:
                    plot_name = plot.replace('_', ' ').replace('.png', '').title()
                    print(f"  • {plot_name}")
                print()

        print(f"Total plots available: {len(plot_files)}")
        print("Access dashboard at: http://localhost:8002/dashboard")
    else:
        print("Plots directory not found")

    print()
    print("=" * 100)
    print("METRICS DISPLAY COMPLETE")
    print("Dashboard: http://localhost:8002/dashboard")
    print("API Docs: http://localhost:8002/docs")
    print("=" * 100)

if __name__ == "__main__":
    display_all_metrics()