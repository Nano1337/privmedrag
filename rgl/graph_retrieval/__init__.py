import os
import sys
import importlib.util

# Get the directory of the current package
current_dir = os.path.dirname(os.path.abspath(__file__))

# List all .so files in the directory
so_files = [f for f in os.listdir(current_dir) if f.endswith('.so')]

# Look for the retrieval module among the .so files
retrieval_module = None
for so_file in so_files:
    if 'retrieval' in so_file.lower():
        # Full path to the .so file
        full_path = os.path.join(current_dir, so_file)
        
        # Always use 'retrieval' as the module name regardless of the actual filename
        module_name = 'retrieval'
        
        # Load the module
        try:
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            retrieval = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = retrieval
            spec.loader.exec_module(retrieval)
            
            # Make it available for import
            globals()['retrieval'] = retrieval
            print(f"Loaded retrieval module from {so_file}")
            break
        except ImportError as e:
            print(f"Failed to load {so_file}: {e}")
            continue

if 'retrieval' not in globals():
    print(f"WARNING: Could not find retrieval module in {current_dir}")
    print(f"Available .so files: {so_files}")
