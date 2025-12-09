"""
Model evaluation script for EMNIST CNN.

This script:
- Loads a trained model from disk
- Evaluates on EMNIST test set
- Computes comprehensive metrics (accuracy, top-5, per-class metrics)
- Saves results to JSON
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import load_emnist
from src.utils.label_mapping import load_label_mapping


def evaluate_model(
    model_path: str,
    output_path: str = "models/evaluation_results.json",
    verbose: bool = True
):
    """
    Evaluate a trained EMNIST model.
    
    Args:
        model_path: Path to saved model file (.keras or .h5)
        output_path: Path to save evaluation results JSON
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if verbose:
        print("="*70)
        print("EMNIST MODEL EVALUATION")
        print("="*70)
    
    # Load model
    if verbose:
        print(f"\n[1/5] Loading model from {model_path}...")
    
    model = keras.models.load_model(model_path)
    
    if verbose:
        print("✓ Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
    
    # Load test data
    if verbose:
        print("\n[2/5] Loading EMNIST test set...")
    
    _, _, x_test, y_test_labels = load_emnist()
    
    # Preprocess test data
    x_test = x_test.astype(np.float32) / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_test = tf.keras.utils.to_categorical(y_test_labels, 62)
    
    if verbose:
        print(f"✓ Test set loaded: {x_test.shape[0]:,} samples")
    
    # Evaluate overall metrics
    if verbose:
        print("\n[3/5] Computing overall metrics...")
    
    test_results = model.evaluate(x_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_top5_accuracy = test_results[2] if len(test_results) > 2 else None
    
    if verbose:
        print(f"✓ Overall metrics computed")
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Test accuracy: {test_accuracy*100:.2f}%")
        if test_top5_accuracy:
            print(f"  Top-5 accuracy: {test_top5_accuracy*100:.2f}%")
    
    # Get predictions
    if verbose:
        print("\n[4/5] Computing per-class metrics...")
    
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    # Load label mapping
    label_mapping = load_label_mapping()
    class_names = [label_mapping[i] for i in range(62)]
    
    # Compute per-class metrics
    report_dict = classification_report(
        y_test_labels, 
        y_pred_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Extract per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        if class_name in report_dict:
            per_class_metrics[class_name] = {
                'precision': float(report_dict[class_name]['precision']),
                'recall': float(report_dict[class_name]['recall']),
                'f1-score': float(report_dict[class_name]['f1-score']),
                'support': int(report_dict[class_name]['support'])
            }
    
    if verbose:
        print(f"✓ Per-class metrics computed for {len(per_class_metrics)} classes")
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    
    # Find commonly confused pairs
    confused_pairs = []
    for i in range(62):
        for j in range(62):
            if i != j and conf_matrix[i, j] > 50:  # Threshold for "commonly confused"
                confused_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(conf_matrix[i, j]),
                    'percentage': float(conf_matrix[i, j] / conf_matrix[i].sum() * 100)
                })
    
    # Sort by count
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    if verbose:
        print(f"  Found {len(confused_pairs)} commonly confused pairs")
    
    # Compile results
    if verbose:
        print("\n[5/5] Saving results...")
    
    results = {
        'model_path': str(model_path),
        'test_set_size': int(x_test.shape[0]),
        'overall_metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_top5_accuracy': float(test_top5_accuracy) if test_top5_accuracy else None
        },
        'aggregate_metrics': {
            'macro_avg': {
                'precision': float(report_dict['macro avg']['precision']),
                'recall': float(report_dict['macro avg']['recall']),
                'f1-score': float(report_dict['macro avg']['f1-score'])
            },
            'weighted_avg': {
                'precision': float(report_dict['weighted avg']['precision']),
                'recall': float(report_dict['weighted avg']['recall']),
                'f1-score': float(report_dict['weighted avg']['f1-score'])
            }
        },
        'per_class_metrics': per_class_metrics,
        'commonly_confused_pairs': confused_pairs[:20],  # Top 20
        'confusion_matrix_shape': list(conf_matrix.shape)
    }
    
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"✓ Results saved to {output_path}")
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        if test_top5_accuracy:
            print(f"Top-5 Accuracy: {test_top5_accuracy*100:.2f}%")
        print(f"Macro F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
        
        print(f"\nTop 5 confused character pairs:")
        for i, pair in enumerate(confused_pairs[:5], 1):
            print(f"  {i}. '{pair['true_class']}' → '{pair['predicted_class']}': "
                  f"{pair['count']} times ({pair['percentage']:.1f}%)")
        
        print("="*70)
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Evaluate trained EMNIST model')
    parser.add_argument('--model', type=str, default='models/emnist_cnn_v1.keras',
                        help='Path to trained model file')
    parser.add_argument('--output', type=str, default='models/evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train a model first or specify a different path.")
        return 1
    
    # Run evaluation
    try:
        evaluate_model(
            model_path=args.model,
            output_path=args.output,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
