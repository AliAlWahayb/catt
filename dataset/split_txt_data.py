import os
import random

# Input and output paths
input_dir = 'dataset/txt'
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# Create output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Find the single txt file in input_dir
all_txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
if not all_txt_files:
    raise FileNotFoundError('No .txt file found in dataset/txt')
input_file = os.path.join(input_dir, all_txt_files[0])

# Read all lines
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.rstrip('\n') for line in f if line.strip()]

random.shuffle(lines)

n_val = max(1, int(0.15 * len(lines)))
n_test = max(1, int(0.15 * len(lines)))
n_train = max(0, len(lines) - n_val - n_test)

val_lines = lines[:n_val]
test_lines = lines[n_val:n_val+n_test]
train_lines = lines[n_val+n_test:]

with open(os.path.join(train_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for l in train_lines:
        f.write(l + '\n')
with open(os.path.join(val_dir, 'val.txt'), 'w', encoding='utf-8') as f:
    for l in val_lines:
        f.write(l + '\n')
with open(os.path.join(test_dir, 'test.txt'), 'w', encoding='utf-8') as f:
    for l in test_lines:
        f.write(l + '\n')

print(f"Train: {len(train_lines)}, Val: {len(val_lines)}, Test: {len(test_lines)}")
print("Done.")
