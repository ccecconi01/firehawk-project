"""
Script to convert CSV to JSON.
Usage: python csv_to_json.py <file.csv> [--output file.json] [--pretty]

Example:
  python csv_to_json.py data/test_latest_30_rows.csv
  python csv_to_json.py data/test_latest_30_rows.csv --output data.json --pretty
"""

import pandas as pd
import json
import sys
import os
from pathlib import Path

def csv_to_json(csv_file, output_file=None, pretty=False):
    """
    Convert CSV to JSON.
    
    Args:
        csv_file: Path to CSV file
        output_file: Path to JSON output file (optional)
        pretty: If True, formats JSON with indentation for readability
    
    Returns:
        str: path to the generated JSON file
    """
    
    # Validate CSV file
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        return None
    
    try:
        # Read CSV
        print(f"→ Reading CSV: {csv_file}")
        df = pd.read_csv(csv_file, dtype={'ID_Incidente': str})
        print(f"  ✓ {len(df)} rows read")
        
        # Convert to JSON
        # orient='records' converts to an array of objects (ideal for APIs)
        json_data = df.to_dict(orient='records')
        
        # Determine output file
        if output_file is None:
            # Auto-generate filename: file.csv → file.json
            csv_path = Path(csv_file)
            output_file = csv_path.parent / f"{csv_path.stem}.json"
        
        output_file = str(output_file)
        
        # Save JSON
        print(f"\n→ Saving JSON: {output_file}")
        
        if pretty:
            # Pretty format with indentation
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"  Pretty-formatted JSON saved successfully")
        else:
            # Compact format (smaller file size)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False)
            print(f"  Compact JSON saved successfully")
        
        # Information about the generated file
        file_size_kb = os.path.getsize(output_file) / 1024
        print(f"\n Conversion completed!")
        print(f"  File: {output_file}")
        print(f"  Size: {file_size_kb:.2f} KB")
        print(f"  Rows: {len(json_data)}")
        
        # Preview of JSON
        print(f"\n Preview (first row):")
        print("-" * 70)
        if json_data:
            print(json.dumps(json_data[0], indent=2, ensure_ascii=False))
        
        return output_file
        
    except Exception as e:
        print(f" ERROR processing file: {e}")
        return None

def main():
    """Main - processes command line arguments."""
    
    print("=" * 70)
    print("  CSV to JSON Converter")
    print("=" * 70)
    
    # Validate arguments
    if len(sys.argv) < 2:
        print("\n USAGE: python csv_to_json.py <file.csv> [--output file.json] [--pretty]")
        print("\nExamples:")
        print("  python csv_to_json.py data/test_latest_30_rows.csv")
        print("  python csv_to_json.py data/test_latest_30_rows.csv --pretty")
        print("  python csv_to_json.py data/test_latest_30_rows.csv --output data.json --pretty")
        print("\nOptions:")
        print("  --output FILE     Define the name of the output JSON file")
        print("  --pretty          Format JSON with indentation (more readable)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = None
    pretty = False
    
    # Parse optional arguments
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--pretty':
            pretty = True
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
    
    # Convert
    result = csv_to_json(csv_file, output_file, pretty)
    
    if result:
        print(f"\n Ready to use!")
        print(f"   JSON file: {result}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
