import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

class VisualizationEngine:
    """
    Visualization tools for student performance analysis
    """
    
    def __init__(self, style='darkgrid'):
        """Initialize visualization engine"""
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_data_distribution(self, df, save_path=None):
        """
        Plot distribution of all features in the dataset
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the figure
        """
        features = df.drop('Performance', axis=1).columns
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            axes[idx].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {feature}')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        # Remove the last empty subplot
        axes[-1].remove()
        
        plt.suptitle('Feature Distributions in Student Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, df, save_path=None):
        """
        Plot correlation matrix heatmap
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(10, 8))
        
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, cbar_kws={'label': 'Correlation'})
        
        plt.title('Correlation Matrix - Student Performance Features', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_distribution(self, df, save_path=None):
        """
        Plot distribution of target variable
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        
        performance_counts = df['Performance'].value_counts()
        colors = ['#ff6b6b', '#51cf66']
        labels = ['Poor Performance', 'Good Performance']
        
        bars = plt.bar(range(len(performance_counts)), performance_counts.values, 
                       color=colors, edgecolor='black', alpha=0.7)
        
        plt.xticks(range(len(performance_counts)), 
                   [labels[i] for i in performance_counts.index])
        plt.ylabel('Number of Students')
        plt.title('Performance Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix from sklearn
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Poor', 'Good'],
                   yticklabels=['Poor', 'Good'],
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
        Plot ROC curve
        
        Args:
            y_test: True labels
            y_pred_proba: Predicted probabilities
            save_path (str): Path to save the figure
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, coefficients, feature_names, save_path=None):
        """
        Plot feature importance based on model coefficients
        
        Args:
            coefficients: Model coefficients
            feature_names (list): Feature names
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        
        # Sort by absolute coefficient value
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_coef = coefficients[sorted_idx]
        
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in sorted_coef]
        
        bars = plt.barh(sorted_features, sorted_coef, color=colors, edgecolor='black', alpha=0.7)
        
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance in Logistic Regression', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {width:.4f}',
                    ha='left' if width > 0 else 'right', va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_distribution(self, predictions_df, save_path=None):
        """
        Plot distribution of predictions and confidence
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            save_path (str): Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predictions pie chart
        pred_counts = predictions_df['Predicted_Performance'].value_counts()
        colors = ['#ff6b6b', '#51cf66']
        axes[0].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        axes[0].set_title('Predicted Performance Distribution', fontsize=12, fontweight='bold')
        
        # Confidence distribution
        axes[1].hist(predictions_df['Confidence'], bins=30, edgecolor='black', 
                    alpha=0.7, color='steelblue')
        axes[1].axvline(predictions_df['Confidence'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {predictions_df['Confidence'].mean():.2f}%")
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
