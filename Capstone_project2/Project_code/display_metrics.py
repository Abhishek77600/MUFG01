import json
import os

def load_metadata(artifacts_dir='artifacts'):
    """Load metadata from artifacts directory"""
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        return json.load(f)

def display_model_metrics():
    """Display metrics for each model and highlight the best one"""
    metadata = load_metadata()
    evaluations = metadata['evaluations']
    chosen_model = metadata['chosen_model']

    print("=" * 80)
    print("HEART DISEASE PREDICTION MODEL METRICS")
    print("=" * 80)
    print(f"Best Model: {chosen_model.upper()} (*BEST*)")
    print()

    # Prepare data for table
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Best?'}")
    print("-" * 90)

    # Sort by ROC-AUC descending
    sorted_models = sorted(evaluations.items(), key=lambda x: x[1]['test_eval']['roc_auc'], reverse=True)

    for model_name, eval_data in sorted_models:
        test_eval = eval_data['test_eval']
        report = test_eval['classification_report']

        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        roc_auc = test_eval['roc_auc']

        is_best = "*YES*" if model_name == chosen_model else "NO"
        model_display = model_name.replace('_', ' ').title()

        print(f"{model_display:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {roc_auc:<10.4f} {is_best}")

    print()

    # Display detailed classification report for best model
    print(f"Detailed Classification Report for Best Model ({chosen_model.upper()}):")
    print("-" * 60)
    best_eval = evaluations[chosen_model]['test_eval']
    report = best_eval['classification_report']

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

def display_available_plots():
    """Display list of available plots"""
    plots_dir = 'plots'
    if os.path.exists(plots_dir):
        print("Available Plots:")
        print("-" * 40)
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        for plot in sorted(plot_files):
            print(f"  - {plot}")
        print()
        print("To view plots, you can open the PNG files in an image viewer.")
        print("Key plots to check:")
        print("  - model_performance_comparison.png (Baseline vs Optimized metrics)")
        print("  - roc_curves_optimized.png (ROC curves for all models)")
        print("  - enhanced_confusion_matrices.png (Confusion matrices)")
        print("  - model_improvement_summary.png (Improvement from baseline)")
    else:
        print("Plots directory not found.")

if __name__ == '__main__':
    display_model_metrics()
    display_available_plots()