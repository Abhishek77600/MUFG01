import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Healthcare color scheme
HEALTH_COLORS = {
    'primary': '#2E86AB',      # Medical blue
    'secondary': '#A23B72',    # Deep pink
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red for disease
    'neutral': '#6B7B8C',      # Gray
    'background': '#F7F9FC'    # Light blue-gray
}

# Load the dataset
df = pd.read_csv('heart_disease_dataset.csv')

print("HEART DISEASE DATASET EXPLORATORY ANALYSIS")
print("=" * 60)

# Dataset Overview Dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Heart Disease Dataset Overview', fontsize=20, fontweight='bold', color=HEALTH_COLORS['primary'])

# 1. Dataset Shape
ax1.text(0.5, 0.7, f'{df.shape[0]}', ha='center', va='center', fontsize=48, fontweight='bold', color=HEALTH_COLORS['primary'])
ax1.text(0.5, 0.3, 'Total Samples', ha='center', va='center', fontsize=16, color=HEALTH_COLORS['neutral'])
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.add_patch(Circle((0.5, 0.5), 0.3, facecolor=HEALTH_COLORS['background'], edgecolor=HEALTH_COLORS['primary'], linewidth=3))

# 2. Features Count
ax2.text(0.5, 0.7, f'{df.shape[1]}', ha='center', va='center', fontsize=48, fontweight='bold', color=HEALTH_COLORS['secondary'])
ax2.text(0.5, 0.3, 'Features', ha='center', va='center', fontsize=16, color=HEALTH_COLORS['neutral'])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.add_patch(Circle((0.5, 0.5), 0.3, facecolor=HEALTH_COLORS['background'], edgecolor=HEALTH_COLORS['secondary'], linewidth=3))

# 3. Class Distribution
class_counts = df['heart_disease'].value_counts()
colors = [HEALTH_COLORS['success'], HEALTH_COLORS['primary']]
ax3.pie(class_counts.values, labels=['No Disease', 'Heart Disease'], autopct='%1.1f%%',
        colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 3})
ax3.set_title('Class Distribution', fontweight='bold', color=HEALTH_COLORS['primary'])

# 4. Missing Values
missing_data = df.isnull().sum()
ax4.bar(range(len(missing_data)), missing_data.values, color=HEALTH_COLORS['accent'])
ax4.set_xticks(range(len(missing_data)))
ax4.set_xticklabels(missing_data.index, rotation=45, ha='right')
ax4.set_title('Missing Values per Feature', fontweight='bold', color=HEALTH_COLORS['primary'])
ax4.set_ylabel('Count')

plt.tight_layout()
plt.savefig('plots/dataset_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# Enhanced Correlation Matrix
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
corr = df.corr()

# Create custom colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})

plt.title('Feature Correlation Matrix', fontsize=18, fontweight='bold', color=HEALTH_COLORS['primary'], pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/enhanced_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Distributions with Health Context
numerical_cols = df.select_dtypes(include=[np.number]).columns[:-1]  # Exclude target

# Create a comprehensive distribution plot
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle('Feature Distributions by Heart Disease Status', fontsize=20, fontweight='bold', color=HEALTH_COLORS['primary'])

axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    if i < len(axes):
        # Create violin plots with box plots inside
        sns.violinplot(data=df, x='heart_disease', y=col, ax=axes[i],
                      palette=[HEALTH_COLORS['primary'], HEALTH_COLORS['success']],
                      inner='box', linewidth=2)

        axes[i].set_title(f'{col.replace("_", " ").title()}', fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Heart Disease' if i >= 9 else '')
        axes[i].grid(True, alpha=0.3)

# Remove empty subplots
for i in range(len(numerical_cols), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('plots/feature_distributions_violin.png', dpi=300, bbox_inches='tight')
plt.close()

# Key Health Metrics Dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Key Health Metrics Analysis', fontsize=20, fontweight='bold', color=HEALTH_COLORS['primary'])

# Age distribution by disease status
sns.histplot(data=df, x='age', hue='heart_disease', ax=ax1,
             palette=[HEALTH_COLORS['primary'], HEALTH_COLORS['success']],
             alpha=0.7, bins=20)
ax1.set_title('Age Distribution by Heart Disease Status', fontweight='bold')
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Count')
ax1.grid(True, alpha=0.3)

# Blood pressure vs cholesterol scatter
sns.scatterplot(data=df, x='resting_blood_pressure', y='cholesterol',
                hue='heart_disease', ax=ax2, s=60,
                palette=[HEALTH_COLORS['primary'], HEALTH_COLORS['success']])
ax2.set_title('Blood Pressure vs Cholesterol', fontweight='bold')
ax2.set_xlabel('Resting Blood Pressure (mm Hg)')
ax2.set_ylabel('Cholesterol (mg/dl)')
ax2.grid(True, alpha=0.3)

# Heart rate analysis
sns.boxplot(data=df, x='heart_disease', y='max_heart_rate', ax=ax3,
            palette=[HEALTH_COLORS['primary'], HEALTH_COLORS['success']])
ax3.set_title('Maximum Heart Rate by Disease Status', fontweight='bold')
ax3.set_xlabel('Heart Disease')
ax3.set_ylabel('Max Heart Rate')
ax3.grid(True, alpha=0.3)

# ST depression analysis
sns.violinplot(data=df, x='heart_disease', y='st_depression', ax=ax4,
               palette=[HEALTH_COLORS['primary'], HEALTH_COLORS['success']],
               inner='quartile')
ax4.set_title('ST Depression by Disease Status', fontweight='bold')
ax4.set_xlabel('Heart Disease')
ax4.set_ylabel('ST Depression')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/health_metrics_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# Statistical Summary Table
print("\nDATASET STATISTICS")
print("-" * 60)
print(f"Total Samples: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print(f"Target Distribution: {class_counts[0]} No Disease ({class_counts[0]/len(df)*100:.1f}%), {class_counts[1]} Heart Disease ({class_counts[1]/len(df)*100:.1f}%)")

# Feature statistics
print("\nFEATURE STATISTICS")
print("-" * 60)
stats_df = df.describe().round(2)
print(stats_df)

# Outlier Analysis with Visualization
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

print("\nOUTLIER ANALYSIS")
print("-" * 60)
outlier_counts = []
for col in numerical_cols:
    count, lower, upper = detect_outliers_iqr(df, col)
    outlier_counts.append(count)
    print(f"{col}: {count} outliers (bounds: {lower:.2f} - {upper:.2f})")

# Outlier summary plot
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(numerical_cols)), outlier_counts, color=HEALTH_COLORS['accent'], alpha=0.7)
plt.xticks(range(len(numerical_cols)), [col.replace('_', '\n') for col in numerical_cols], rotation=45, ha='right')
plt.title('Outlier Count by Feature', fontsize=16, fontweight='bold', color=HEALTH_COLORS['primary'])
plt.ylabel('Number of Outliers')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, count in zip(bars, outlier_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nEDA COMPLETED - {len(numerical_cols) + 4} visualizations saved to plots/ directory")
print("Generated files:")
print("  - dataset_overview.png")
print("  - enhanced_correlation_heatmap.png")
print("  - feature_distributions_violin.png")
print("  - health_metrics_dashboard.png")
print("  - outlier_analysis.png")
print("  - Individual histograms and boxplots for each feature")