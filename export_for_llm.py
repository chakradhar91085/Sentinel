import os

EXTENSIONS = ('.py', '.html', '.css', '.js', '.md', '.txt', 'Dockerfile')
IGNORE_DIRS = ('.git', '__pycache__', 'sentinel_model', 'frontend_old')

output_file = 'sentinel_full_codebase.txt'

with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        # Exclude ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file.endswith(EXTENSIONS) or 'Dockerfile' in file:
                filepath = os.path.join(root, file)
                
                # Skip the output file and this script itself
                if file == output_file or file == 'export_for_llm.py': 
                    continue
                
                # Write file header
                outfile.write(f"\n\n{'='*80}\n")
                outfile.write(f"File: {filepath}\n")
                outfile.write(f"{'='*80}\n\n")
                
                # Write file contents
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")

print(f"Successfully bundled codebase into {output_file}")
