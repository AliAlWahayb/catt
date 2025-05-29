import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CONFIG
LOGS_DIR = 'catt_ed_model_v1/logs'  # Change if using EO or different dir
CHECKPOINT_DIR = 'catt_ed_model_v1/'

# 1. Plot loss/accuracy curves from CSVLogger logs
def plot_training_curves(logs_dir=LOGS_DIR):
    csv_files = glob.glob(os.path.join(logs_dir, '*', 'metrics.csv'))
    if not csv_files:
        print('No metrics.csv found!')
        return
    df = pd.concat([pd.read_csv(f) for f in csv_files])
    for metric in ['train_loss', 'val_loss', 'val_der', 'val_wer', 'val_cer']:
        if metric in df.columns:
            plt.figure()
            plt.plot(df['epoch'], df[metric], label=metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} over epochs')
            plt.legend()
            plt.grid(True)
            plt.show()

# 2. Show confusion matrix (requires true/pred labels)
def plot_confusion_matrix(true_labels, pred_labels, class_names=None):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# 3. Analyze per-class DER (Diacritic Error Rate)
def per_class_der(true_labels, pred_labels, class_names=None):
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))
    der_per_class = 1 - np.diag(cm) / cm.sum(axis=1)
    for idx, der in enumerate(der_per_class):
        print(f'Class {class_names[idx] if class_names else idx}: DER={der:.4f}')
    plt.figure()
    plt.bar(class_names if class_names else range(len(der_per_class)), der_per_class)
    plt.ylabel('DER')
    plt.title('Per-Class DER')
    plt.show()

# 4. Visualize attention weights (if available)
def visualize_attention_weights(model_ckpt_path, input_tensor):
    # This is a placeholder. Actual implementation depends on model internals.
    model = torch.load(model_ckpt_path, map_location='cpu')
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'get_attention_weights'):
        attn_weights = model.transformer.get_attention_weights(input_tensor)
        for i, attn in enumerate(attn_weights):
            plt.figure(figsize=(8, 6))
            sns.heatmap(attn, cmap='viridis')
            plt.title(f'Attention Head {i}')
            plt.show()
    else:
        print('Attention visualization not implemented for this model.')

# 5. Automatically extract and analyze validation predictions if available

def auto_val_analysis():
    # Try to find validation ground truth and predictions in benchmarking dir
    gt_files = glob.glob('benchmarking/all_models_CATT_data/*_gt.txt')
    pred_files = [f for f in glob.glob('benchmarking/all_models_CATT_data/*.txt') if '_gt' not in f]
    if not gt_files or not pred_files:
        print('No ground truth or prediction files found for auto analysis.')
        return
    for gt_file in gt_files:
        gt = [line.strip() for line in open(gt_file, encoding='utf-8') if line.strip()]
        for pred_file in pred_files:
            pred = [line.strip() for line in open(pred_file, encoding='utf-8') if line.strip()]
            if len(gt) != len(pred):
                continue
            print(f'Analyzing: {os.path.basename(pred_file)} vs {os.path.basename(gt_file)}')
            # For confusion matrix and DER, treat each character as a class
            true_labels = list(''.join(gt))
            pred_labels = list(''.join(pred))
            all_classes = sorted(list(set(true_labels + pred_labels)))
            plot_confusion_matrix(true_labels, pred_labels, class_names=all_classes)
            per_class_der(true_labels, pred_labels, class_names=all_classes)
            # Error analysis: save mispredicted samples (line and char level)
            mispred_path = f"error_analysis_{os.path.basename(pred_file).replace('.txt','')}_vs_{os.path.basename(gt_file).replace('.txt','')}.tsv"
            with open(mispred_path, 'w', encoding='utf-8') as fout:
                fout.write('idx\tground_truth\tprediction\tchar_errors\n')
                for idx, (gt_line, pred_line) in enumerate(zip(gt, pred)):
                    if gt_line != pred_line:
                        # Find character-level mismatches
                        char_errors = []
                        min_len = min(len(gt_line), len(pred_line))
                        for i in range(min_len):
                            if gt_line[i] != pred_line[i]:
                                char_errors.append(f'{i}:{gt_line[i]}!={pred_line[i]}')
                        # If lengths differ, note extra chars
                        if len(gt_line) > min_len:
                            char_errors.append(f'{min_len}:gt_extra:{gt_line[min_len:]})')
                        if len(pred_line) > min_len:
                            char_errors.append(f'{min_len}:pred_extra:{pred_line[min_len:]})')
                        fout.write(f'{idx}\t{gt_line}\t{pred_line}\t{"; ".join(char_errors)}\n')
            print(f'Mispredicted samples saved to: {mispred_path}')

if __name__ == '__main__':
    # 1. Plot training/validation curves
    plot_training_curves()
    # 2. Automatically analyze validation predictions if available
    auto_val_analysis()
    # 3. For attention visualization, provide a checkpoint and input
    # visualize_attention_weights('catt_ed_model_v1/last.ckpt', input_tensor)
    pass
