import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Set up professional plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Healthcare color scheme
HEALTH_COLORS = {
    'primary': '#2E86AB',      # Medical blue
    'secondary': '#A23B72',    # Deep pink
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red for disease
    'neutral': '#6B7B8C',      # Gray
    'background': '#F7F9FC'    # Light blue-gray
}

def main():
    # Load the best model and scaler from artifacts directory
    artifacts_dir = 'artifacts'
    model_path = os.path.join(artifacts_dir, 'best_model.pkl')
    scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded model: {metadata.get('chosen_model', 'unknown')}")

    # Load the dataset to get feature names
    data_path = 'heart_disease_dataset.csv'
    df = pd.read_csv(data_path)
    feature_names = df.drop(columns=['heart_disease']).columns.tolist()

    # Prepare data for feature selection
    X = df.drop(columns=['heart_disease'])
    y = df['heart_disease']
    X_scaled = scaler.transform(X)

    # Determine model type and get feature importances
    model_name = metadata['chosen_model']
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        method = 'built-in feature_importances_'
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
        method = 'coefficients'
    else:
        # For other models like SVM, use permutation importance
        # Need to split data for permutation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
        method = 'permutation importance'

    print(f"Using {method} for feature importance analysis.")

    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nFEATURE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Model Type: {model_name.capitalize()}")
    print(f"Importance Method: {method}")
    print("\nTop 10 Feature Importances:")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")

    # Enhanced Feature Importance Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Feature Analysis Dashboard', fontsize=20, fontweight='bold', color=HEALTH_COLORS['primary'])

    # 1. Feature Importance Bar Chart
    top_features = feature_importance_df.head(10)
    bars = ax1.barh(range(len(top_features)), top_features['importance'],
                   color=HEALTH_COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1)

    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([f.replace('_', '\n') for f in top_features['feature']], fontsize=10)
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold', color=HEALTH_COLORS['primary'])
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        ax1.text(importance + 0.001, i, f'{importance:.3f}', va='center', fontsize=9, fontweight='bold')

    # 2. Feature Correlation with Target
    correlations = []
    for feature in feature_names:
        corr = df[feature].corr(df['heart_disease'])
        correlations.append(abs(corr))  # Use absolute correlation

    corr_df = pd.DataFrame({
        'feature': feature_names,
        'correlation': correlations
    }).sort_values('correlation', ascending=True)

    bars2 = ax2.barh(range(len(corr_df)), corr_df['correlation'],
                    color=HEALTH_COLORS['secondary'], alpha=0.8, edgecolor='black', linewidth=1)

    ax2.set_yticks(range(len(corr_df)))
    ax2.set_yticklabels([f.replace('_', '\n') for f in corr_df['feature']], fontsize=9)
    ax2.set_xlabel('Absolute Correlation with Target', fontsize=12, fontweight='bold')
    ax2.set_title('Feature-Target Correlations', fontsize=14, fontweight='bold', color=HEALTH_COLORS['primary'])
    ax2.grid(True, alpha=0.3)

    # 3. Feature Importance vs Correlation
    importance_corr_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'correlation': [abs(df[f].corr(df['heart_disease'])) for f in feature_names]
    })

    scatter = ax3.scatter(importance_corr_df['correlation'], importance_corr_df['importance'],
                         s=100, c=HEALTH_COLORS['accent'], alpha=0.7, edgecolors='black', linewidth=1)

    # Add feature labels for top features
    top_importance_features = importance_corr_df.nlargest(5, 'importance')
    for _, row in top_importance_features.iterrows():
        ax3.annotate(row['feature'].replace('_', '\n'), (row['correlation'], row['importance']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

    ax3.set_xlabel('Absolute Correlation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
    ax3.set_title('Importance vs Correlation', fontsize=14, fontweight='bold', color=HEALTH_COLORS['primary'])
    ax3.grid(True, alpha=0.3)

    # 4. Feature Selection Results
    k = 7
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    selected_scores = selector.scores_[selected_mask]

    selection_df = pd.DataFrame({
        'feature': selected_features,
        'f_score': selected_scores
    }).sort_values('f_score', ascending=True)

    bars4 = ax4.barh(range(len(selection_df)), selection_df['f_score'],
                    color=HEALTH_COLORS['success'], alpha=0.8, edgecolor='black', linewidth=1)

    ax4.set_yticks(range(len(selection_df)))
    ax4.set_yticklabels([f.replace('_', '\n') for f in selection_df['feature']], fontsize=9)
    ax4.set_xlabel('F-Statistic Score', fontsize=12, fontweight='bold')
    ax4.set_title(f'Top {k} Features (SelectKBest)', fontsize=14, fontweight='bold', color=HEALTH_COLORS['primary'])
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars4, selection_df['f_score'])):
        ax4.text(score + 0.5, i, f'{score:.1f}', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/feature_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Individual Feature Analysis Plots
    print("\nGenerating individual feature analysis plots...")

    # Top 5 features detailed analysis
    top_5_features = feature_importance_df['feature'].head(5).tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Top 5 Features: Detailed Analysis', fontsize=20, fontweight='bold', color=HEALTH_COLORS['primary'])

    axes = axes.ravel()

    for i, feature in enumerate(top_5_features):
        if i < len(axes):
            # Create violin plot with box plot inside
            sns.violinplot(data=df, x='heart_disease', y=feature, ax=axes[i],
                          palette=[HEALTH_COLORS['primary'], HEALTH_COLORS['success']],
                          inner='box', linewidth=2)

            axes[i].set_title(f'{feature.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Heart Disease' if i >= 3 else '')
            axes[i].grid(True, alpha=0.3)

    # Remove empty subplot
    if len(top_5_features) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('plots/top_features_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Feature Importance Waterfall Chart
    plt.figure(figsize=(14, 10))

    # Sort features by importance (descending)
    sorted_features = feature_importance_df.sort_values('importance', ascending=False)

    # Create waterfall chart
    cumulative = 0
    x_pos = []
    y_pos = []
    colors = []

    for i, (_, row) in enumerate(sorted_features.iterrows()):
        x_pos.append(i)
        y_pos.append(row['importance'])
        colors.append(HEALTH_COLORS['primary'] if i < 5 else HEALTH_COLORS['neutral'])

    bars = plt.bar(x_pos, y_pos, color=colors, alpha=0.8, edgecolor='black', linewidth=1, width=0.6)

    plt.xticks(x_pos, [f.replace('_', '\n') for f in sorted_features['feature']],
              rotation=45, ha='right', fontsize=10)
    plt.xlabel('Features', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=14, fontweight='bold')
    plt.title('Feature Importance Waterfall', fontsize=18, fontweight='bold', color=HEALTH_COLORS['primary'], pad=20)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, importance in zip(bars, sorted_features['importance']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importance:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/feature_importance_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Enhanced feature analysis visualizations saved:")
    print("  - feature_analysis_dashboard.png")
    print("  - top_features_detailed.png")
    print("  - feature_importance_waterfall.png")

    # Feature selection using SelectKBest with f_classif
    k = 7  # Select top 7 features
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    # Get selected features and their scores
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    selected_scores = selector.scores_[selected_mask]

    print(f"\nTop {k} Selected Features using SelectKBest (f_classif):")
    for feature, score in zip(selected_features, selected_scores):
        print(f"{feature}: {score:.4f}")

    # Summary
    top_features = feature_importance_df['feature'].head(5).tolist()
    print(f"\nTop 5 important features: {', '.join(top_features)}")
    print(f"Selected features for modeling: {', '.join(selected_features)}")

if __name__ == '__main__':
    main()