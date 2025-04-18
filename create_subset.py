import hashlib
import os
from pathlib import Path

pdf_dir = Path('data/lecture/pdf_reports')
csv_path = Path('data/lecture/subset.csv')
csv_content = 'sha1,source_id\n'  # Changed 'company_name' to 'source_id'
count = 0

print(f"Looking for PDFs in: {pdf_dir.resolve()}")

if not pdf_dir.exists():
    print(f"Error: PDF directory not found: {pdf_dir}")
else:
    pdf_files = list(pdf_dir.glob('*.pdf'))
    if not pdf_files:
        print(f"Warning: No PDF files found in {pdf_dir}")
    else:
        print(f"Found {len(pdf_files)} PDF files. Processing...")
        for pdf_file in pdf_files:
            try:
                sha1_hash = hashlib.sha1(pdf_file.read_bytes()).hexdigest()
                # Use filename without extension as source_id
                source_id = pdf_file.stem  # Changed variable name from company_name to source_id
                csv_content += f'{sha1_hash},"{source_id}"\n'  # Add quotes around source_id just in case
                count += 1
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")

        try:
            csv_path.write_text(csv_content, encoding='utf-8')
            print(f"Successfully created {csv_path.resolve()} with {count} entries using 'source_id' column.")
        except Exception as e:
            print(f"Error writing {csv_path}: {e}")
