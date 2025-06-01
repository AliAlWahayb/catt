import os
import glob
import pandas as pd
from tashkeel_tokenizer import TashkeelTokenizer

# Folder containing CSV files
data_dir = 'dataset/csv/clean'
# Output folder for cleaned txt files
output_dir = 'dataset/train/'
os.makedirs(output_dir, exist_ok=True)

tokenizer = TashkeelTokenizer()

csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
all_clean_lines = []

for csv_file in csv_files:
    print(f'Processing: {csv_file}')
    df = pd.read_csv(csv_file)
    # Try to find a column with tashkeel text
    for col in ['text_with_harakat', 'text', 'sentence']:
        if col in df.columns:
            text_col = col
            break
    else:
        print(f'No suitable text column found in {csv_file}, skipping.')
        continue
    clean_lines = []
    for line in df[text_col]:
        clean_line = line
        if clean_line:
            clean_lines.append(clean_line)
    # Save per-csv txt file
    base = os.path.splitext(os.path.basename(csv_file))[0]
    out_path = os.path.join(output_dir, f'{base}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for l in clean_lines:
            f.write(l + '\n')
    all_clean_lines.extend(clean_lines)
    print(f'Saved {len(clean_lines)} lines to {out_path}')

# Optionally, save all lines to a combined file
combined_path = os.path.join(output_dir, 'all_data.txt')
with open(combined_path, 'w', encoding='utf-8') as f:
    for l in all_clean_lines:
        f.write(l + '\n')
print(f'Combined all cleaned lines into {combined_path}')

print('Done! Now run your training script as usual.')
