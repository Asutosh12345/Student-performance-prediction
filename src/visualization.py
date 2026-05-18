import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd


class VisualizationEngine:
    """
    Visualization tools for student performance analysis.
    """

    def __init__(self, style='darkgrid'):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_data_distribution(self, df, save_path=None):
        plot_cols = [c for c in df.columns if c not in ('Performance', 'Student_ID', 'Student_Name')]
        n = len(plot_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = axes.flatten()

        for idx, feature in enumerate(plot_cols):
            axes[idx].hist(df[feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {feature}')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

        for idx in range(len(plot_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Feature Distributions in Student Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_correlation_matrix(self, df, save_path=None):
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix - Student Performance Features',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_performance_distribution(self, df, save_path=None):
        plt.figure(figsize=(10, 6))
        performance_counts = df['Performance'].value_counts().sort_index()

        # FIX 9: Support 3-class labels (0=Poor, 1=Average, 2=Good).
        # Old code only had ['Poor Performance', 'Good Performance'] for index 0 and 1,
        # so a value of 2 would raise an IndexError.
        label_map = {0: 'Poor', 1: 'Average', 2: 'Good'}
        colors    = {0: '#ff6b6b', 1: '#ffd93d', 2: '#51cf66'}

        bar_labels = [label_map.get(i, str(i)) for i in performance_counts.index]
        bar_colors = [colors.get(i, 'steelblue')   for i in performance_counts.index]

        bars = plt.bar(range(len(performance_counts)), performance_counts.values,
                       color=bar_colors, edgecolor='black', alpha=0.8)
        plt.xticks(range(len(performance_counts)), bar_labels)
        plt.ylabel('Number of Students')
        plt.title('Performance Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_confusion_matrix(self, cm, save_path=None):
        # FIX 10: Dynamically size labels to the actual confusion matrix dimensions
        # instead of hardcoding ['Poor', 'Good'] for a 2-class problem.
        # A 3-class model produced a 3×3 matrix but the old code only labelled 2 axes.
        n = cm.shape[0]
        label_map = {0: 'Poor', 1: 'Average', 2: 'Good'}
        tick_labels = [label_map.get(i, f'Class {i}') for i in range(n)]

        plt.figure(figsize=(max(6, n * 2), max(5, n * 2)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tick_labels,
                    yticklabels=tick_labels,
                    cbar_kws={'label': 'Count'})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """
        Plot ROC curves.  Supports both binary and multiclass (one-vs-rest).
        FIX 11: Old code called roc_curve(y_test, y_pred_proba) and passed the
        full 2-D probability matrix as the second argument.  roc_curve expects a
        1-D array; with a matrix it silently used only one column or raised an
        error for 3-class models.  Now we plot one curve per class (OvR).
        """
        plt.figure(figsize=(8, 6))

        n_classes = y_pred_proba.shape[1] if y_pred_proba.ndim == 2 else 1
        label_map = {0: 'Poor', 1: 'Average', 2: 'Good'}
        colors    = ['darkorange', 'steelblue', 'green', 'red']

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            y_test_arr = np.asarray(y_test)
            for cls in range(n_classes):
                fpr, tpr, _ = roc_curve((y_test_arr == cls).astype(int),
                                        y_pred_proba[:, cls])
                roc_auc = auc(fpr, tpr)
                lbl = label_map.get(cls, f'Class {cls}')
                plt.plot(fpr, tpr, color=colors[cls % len(colors)], lw=2,
                         label=f'{lbl} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_feature_importance(self, coefficients, feature_names, save_path=None):
        """
        Plot feature importance.
        FIX 12: For multinomial LR, coef_ is (n_classes, n_features).
        Old code used coef_[0] — only the first class's coefficients — giving a
        misleading picture.  Now we show mean absolute coefficient across classes.
        """
        if coefficients.ndim == 2:
            # Average absolute importance across all classes
            coef_1d = np.mean(np.abs(coefficients), axis=0)
        else:
            coef_1d = coefficients

        sorted_idx      = np.argsort(np.abs(coef_1d))[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_coef     = coef_1d[sorted_idx]

        plt.figure(figsize=(10, 6))
        colors = ['#2ecc71' if c >= 0 else '#e74c3c' for c in sorted_coef]
        bars = plt.barh(sorted_features, sorted_coef, color=colors,
                        edgecolor='black', alpha=0.7)
        plt.xlabel('Mean |Coefficient| across classes')
        plt.title('Feature Importance in Logistic Regression',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2.,
                     f' {width:.4f}',
                     ha='left' if width >= 0 else 'right', va='center')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_predictions_distribution(self, predictions_df, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pred_counts = predictions_df['Predicted_Performance'].value_counts()
        color_map = {'Poor': '#ff6b6b', 'Average': '#ffd93d', 'Good': '#51cf66'}
        pie_colors = [color_map.get(l, 'steelblue') for l in pred_counts.index]

        axes[0].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%',
                    colors=pie_colors, startangle=90,
                    textprops={'fontsize': 12, 'weight': 'bold'})
        axes[0].set_title('Predicted Performance Distribution',
                          fontsize=12, fontweight='bold')

        axes[1].hist(predictions_df['Confidence'], bins=30, edgecolor='black',
                     alpha=0.7, color='steelblue')
        mean_conf = predictions_df['Confidence'].mean()
        axes[1].axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_conf:.2f}%')
        axes[1].set_xlabel('Confidence (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()