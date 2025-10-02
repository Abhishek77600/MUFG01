import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg

# Set up professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Healthcare color scheme
HEALTH_COLORS = {
    'primary': '#2E86AB',      # Medical blue
    'secondary': '#A23B72',    # Deep pink
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red for disease
    'neutral': '#6B7B8C',      # Gray
    'background': '#F7F9FC',   # Light blue-gray
    'models': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # For different models
}

def load_metadata(artifacts_dir='artifacts'):
    """Load metadata from artifacts directory"""
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        return json.load(f)

def create_model_dashboard():
    """Create an attractive dashboard showing all model metrics and results"""
    metadata = load_metadata()
    evaluations = metadata['evaluations']
    chosen_model = metadata['chosen_model']

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('HEART DISEASE PREDICTION MODEL DASHBOARD', fontsize=24, fontweight='bold',
                color=HEALTH_COLORS['primary'], y=0.98)

    # Add subtitle
    fig.text(0.5, 0.95, f'Best Model: {chosen_model.replace("_", " ").title()} (ROC-AUC: {evaluations[chosen_model]["test_eval"]["roc_auc"]:.4f})',
             ha='center', fontsize=16, style='italic', color=HEALTH_COLORS['secondary'])

    # 1. Model Performance Comparison Table (Text-based in subplot)
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2)
    ax1.axis('off')

    # Prepare data
    models = list(evaluations.keys())
    metrics_data = []

    for model in models:
        eval_data = evaluations[model]['test_eval']
        report = eval_data['classification_report']
        metrics_data.append({
            'Model': model.replace('_', ' ').title(),
            'Accuracy': report['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score'],
            'ROC-AUC': eval_data['roc_auc'],
            'Best': model == chosen_model
        })

    # Sort by ROC-AUC
    metrics_data.sort(key=lambda x: x['ROC-AUC'], reverse=True)

    # Create table
    table_data = []
    for i, data in enumerate(metrics_data):
        best_marker = " ★" if data['Best'] else ""
        table_data.append([
            f"{data['Model']}{best_marker}",
            f"{data['Accuracy']:.4f}",
            f"{data['Precision']:.4f}",
            f"{data['Recall']:.4f}",
            f"{data['F1-Score']:.4f}",
            f"{data['ROC-AUC']:.4f}"
        ])

    table = ax1.table(cellText=table_data,
                      colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                      loc='center',
                      cellLoc='center',
                      colColours=[HEALTH_COLORS['primary']] * 6,
                      cellColours=[[HEALTH_COLORS['background'] if not row[0].endswith('★') else ['#FFF3CD']*6 for row in table_data][i]] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax1.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)

    # 2. ROC-AUC Comparison Bar Chart
    ax2 = plt.subplot2grid((4, 3), (0, 2))
    model_names = [d['Model'] for d in metrics_data]
    roc_scores = [d['ROC-AUC'] for d in metrics_data]
    colors = [HEALTH_COLORS['success'] if d['Best'] else HEALTH_COLORS['neutral'] for d in metrics_data]

    bars = ax2.bar(range(len(model_names)), roc_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, roc_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Confusion Matrix for Best Model
    ax3 = plt.subplot2grid((4, 3), (1, 0))
    best_eval = evaluations[chosen_model]['test_eval']
    report = best_eval['classification_report']

    # Extract confusion matrix values (approximated from classification report)
    # Since we don't have actual CM, we'll create a visual representation
    cm_data = np.array([
        [report['0']['support'] * report['0']['recall'], report['0']['support'] * (1 - report['0']['recall'])],
        [report['1']['support'] * (1 - report['1']['recall']), report['1']['support'] * report['1']['recall']]
    ])

    sns.heatmap(cm_data, annot=True, fmt='.1f', cmap='Blues', ax=ax3, cbar=True, square=True)
    ax3.set_title(f'Confusion Matrix\n{chosen_model.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    ax3.set_xticklabels(['No Disease', 'Heart Disease'])
    ax3.set_yticklabels(['No Disease', 'Heart Disease'])

    # 4. Performance Radar Chart
    ax4 = plt.subplot2grid((4, 3), (1, 1), projection='polar')

    # Prepare data for radar chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    for i, model_data in enumerate(metrics_data[:3]):  # Top 3 models
        values = [model_data[cat] for cat in categories]
        values += values[:1]  # Close the circle

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        color = HEALTH_COLORS['models'][i] if not model_data['Best'] else HEALTH_COLORS['success']
        linewidth = 3 if model_data['Best'] else 2
        alpha = 1.0 if model_data['Best'] else 0.7

        ax.plot(angles, values, 'o-', linewidth=linewidth, label=model_data['Model'],
               color=color, alpha=alpha)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Comparison (Top 3 Models)', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)

    # Move to correct position
    ax4.figure = fig
    ax4.set_position(ax.get_position())
    plt.close(fig)  # Close the temporary figure

    # 5. Key Metrics Summary
    ax5 = plt.subplot2grid((4, 3), (1, 2))
    ax5.axis('off')

    best_metrics = metrics_data[0]  # First one is best (sorted)
    summary_text = f"""
    BEST MODEL SUMMARY
    {'─' * 30}

    Model: {best_metrics['Model']}
    ROC-AUC: {best_metrics['ROC-AUC']:.4f}
    Accuracy: {best_metrics['Accuracy']:.4f}
    Precision: {best_metrics['Precision']:.4f}
    Recall: {best_metrics['Recall']:.4f}
    F1-Score: {best_metrics['F1-Score']:.4f}

    Best Parameters:
    {json.dumps(evaluations[chosen_model]['best_params'], indent=2)}
    """

    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=HEALTH_COLORS['background'],
                      edgecolor=HEALTH_COLORS['primary']))

    # 6. Available Plots Gallery
    ax6 = plt.subplot2grid((4, 3), (2, 0), colspan=3)
    ax6.axis('off')

    plots_dir = 'plots'
    key_plots = [
        'model_performance_comparison.png',
        'roc_curves_optimized.png',
        'enhanced_confusion_matrices.png',
        'model_improvement_summary.png'
    ]

    ax6.text(0.5, 0.95, 'KEY VISUALIZATIONS AVAILABLE', ha='center', fontsize=16, fontweight='bold',
             color=HEALTH_COLORS['primary'], transform=ax6.transAxes)

    plot_info = """
    Model Performance Comparison    ROC Curves Optimized    Confusion Matrices    Model Improvement Summary
    Feature Importance Charts      Correlation Heatmaps    Distribution Plots    Outlier Analysis
    """

    ax6.text(0.5, 0.7, plot_info, ha='center', fontsize=12, transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=HEALTH_COLORS['background'],
                      edgecolor=HEALTH_COLORS['secondary']))

    ax6.text(0.5, 0.3, f"All plots saved in '{plots_dir}/' directory\nOpen PNG files to view detailed visualizations",
             ha='center', fontsize=10, style='italic', transform=ax6.transAxes)

    # 7. Model Rankings
    ax7 = plt.subplot2grid((4, 3), (3, 0), colspan=2)

    y_pos = np.arange(len(metrics_data))
    performance_scores = [d['ROC-AUC'] for d in metrics_data]

    bars = ax7.barh(y_pos, performance_scores, color=[HEALTH_COLORS['success'] if d['Best'] else HEALTH_COLORS['neutral'] for d in metrics_data],
                    alpha=0.8, edgecolor='black', linewidth=1)

    ax7.set_yticks(y_pos)
    ax7.set_yticklabels([d['Model'] for d in metrics_data])
    ax7.set_xlabel('ROC-AUC Score')
    ax7.set_title('Model Rankings (by ROC-AUC)', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, performance_scores)):
        ax7.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # 8. Project Summary
    ax8 = plt.subplot2grid((4, 3), (3, 2))
    ax8.axis('off')

    summary = f"""
    PROJECT SUMMARY
    {'─' * 20}

    • Dataset: Heart Disease Prediction
    • Models Trained: {len(evaluations)}
    • Best Model: {chosen_model.replace('_', ' ').title()}
    • Test ROC-AUC: {evaluations[chosen_model]['test_eval']['roc_auc']:.4f}
    • CV ROC-AUC: {evaluations[chosen_model]['cv_score']:.4f}

    • Features Engineered: ✓
    • SMOTE Applied: ✓
    • Hyperparameter Optimization: ✓
    • Ensemble Models: ✓

    Generated {len([f for f in os.listdir(plots_dir) if f.endswith('.png')]) if os.path.exists(plots_dir) else 0} visualizations
    """

    ax8.text(0.1, 0.9, summary, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=HEALTH_COLORS['background'],
                      edgecolor=HEALTH_COLORS['accent']))

    plt.tight_layout()
    plt.savefig('model_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Dashboard saved as 'model_dashboard.png'")
    print("Open the PNG file to view the comprehensive model performance dashboard.")

if __name__ == '__main__':
    create_model_dashboard()