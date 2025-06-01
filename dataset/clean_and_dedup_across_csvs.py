import re
import os
import sys
import pandas as pd

# Usage: python clean_and_dedup_across_csvs.py <csv_folder>

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.strip()
    text = text.replace('ٱ', 'ا')
    text = text.replace('\u0640', '')
    text = ' '.join(re.sub(u"[^\u0621-\u063A\u0640-\u0652\u0670\u0671\ufefb\ufef7\ufef5\ufef9 ]", " ", text,  flags=re.UNICODE).split())
    if len(text) < 5:
        return ''
    if not re.search(r'[\u064b-\u0652]', text):
        return ''
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_and_dedup_across_csvs.py <csv_folder>")
        sys.exit(1)
    csv_folder = sys.argv[1]
    out_dir = os.path.join(csv_folder, 'clean')
    os.makedirs(out_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    if not csv_files:
        print(f'No CSV files found in {csv_folder}')
        sys.exit(1)
    # Track which file each duplicate would have gone to
    duplicate_counts = {csv_file: 0 for csv_file in csv_files}
    seen = set()
    file_to_lines = {csv_file: [] for csv_file in csv_files}
    for csv_file in csv_files:
        input_csv = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(input_csv)
        for col in ['text_with_harakat', 'text_with_tashkeel', 'text', 'sentence']:
            if col in df.columns:
                text_col = col
                break
        else:
            print(f'Skipping {csv_file}: no suitable text column found.')
            continue
        for line in df[text_col]:
            # Split the original line on newlines BEFORE cleaning, to preserve verses/lines
            for raw_subline in str(line).splitlines():
                cleaned = clean_text(raw_subline)
                if cleaned:
                    if cleaned not in seen:
                        seen.add(cleaned)
                        file_to_lines[csv_file].append(cleaned)

    # Write cleaned, deduplicated lines for each file
    for csv_file, lines in file_to_lines.items():
        if not lines:
            continue
        out_path = os.path.join(out_dir, f"{os.path.splitext(csv_file)[0]}_cleaned.csv")
        pd.Series(lines).to_csv(out_path, index=False, header=['text_with_harakat'], encoding='utf-8')
        print(f'Saved cleaned file to {out_path}')
    # Print feedback summary
    for csv_file, count in duplicate_counts.items():
        if count > 0:
            print(f'In file {csv_file} there are {count} lines that are duplicates from another file.')
    print('All files processed!')
