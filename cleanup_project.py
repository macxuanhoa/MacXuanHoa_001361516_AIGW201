import os
import shutil
from pathlib import Path

def delete_file(filepath):
    """Safely delete a file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted: {filepath}")
            return True
    except Exception as e:
        print(f"Error deleting {filepath}: {e}")
    return False

def create_directory(dir_path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {dir_path}: {e}")
        return False

def move_file(src, dest_dir):
    """Move a file to a destination directory."""
    try:
        if os.path.exists(src):
            dest = os.path.join(dest_dir, os.path.basename(src))
            shutil.move(src, dest)
            print(f"Moved: {src} -> {dest}")
            return True
    except Exception as e:
        print(f"Error moving {src}: {e}")
    return False

def check_file_imports():
    """Check which of tools.py or ml_tools.py is being imported in app.py and assistant.py."""
    tools_imported = False
    ml_tools_imported = False
    
    for filename in ['app.py', 'assistant.py']:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'import tools' in content or 'from tools import' in content:
                tools_imported = True
            if 'import ml_tools' in content or 'from ml_tools import' in content:
                ml_tools_imported = True
    
    return tools_imported, ml_tools_imported

def remove_pycache_dirs():
    """Recursively remove all __pycache__ directories."""
    removed = 0
    for root, dirs, _ in os.walk('.', topdown=False):
        if '__pycache__' in dirs:
            dir_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(dir_path)
                print(f"Removed: {dir_path}")
                removed += 1
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")
    return removed

def print_directory_tree():
    """Print the current directory structure."""
    print("\nCurrent directory structure:")
    print(".")
    
    def print_tree(start_path, prefix=''):
        try:
            items = sorted(os.listdir(start_path))
            for i, item in enumerate(items):
                path = os.path.join(start_path, item)
                if os.path.isdir(path):
                    if i == len(items) - 1:
                        print(f"{prefix}└── {item}/")
                        print_tree(path, prefix + "    ")
                    else:
                        print(f"{prefix}├── {item}/")
                        print_tree(path, prefix + "│   ")
                else:
                    if i == len(items) - 1:
                        print(f"{prefix}└── {item}")
                    else:
                        print(f"{prefix}├── {item}")
        except Exception as e:
            print(f"Error reading directory {start_path}: {e}")
    
    print_tree('.')

def main():
    print("Starting project cleanup...\n")
    
    # 1. Handle Duplicate/Redundant Files
    # Delete train_model.py if train.py exists
    if os.path.exists('train.py') and os.path.exists('train_model.py'):
        delete_file('train_model.py')
    
    # Check which tools file is being used
    tools_used, ml_tools_used = check_file_imports()
    if tools_used and ml_tools_used:
        print("WARNING: Both tools.py and ml_tools.py are being used in the project. Not deleting either.")
    elif tools_used:
        delete_file('ml_tools.py')
    elif ml_tools_used:
        delete_file('tools.py')
        # Rename ml_tools.py to tools.py if it exists
        if os.path.exists('ml_tools.py'):
            os.rename('ml_tools.py', 'tools.py')
            print("Renamed: ml_tools.py -> tools.py")
    
    # Delete app.log
    delete_file('app.log')
    
    # 2. Organize Development Scripts
    # Create scripts/ directory and move create_housing_data.py
    if create_directory('scripts'):
        if os.path.exists('create_housing_data.py'):
            move_file('create_housing_data.py', 'scripts')
    
    # Create tests/ directory and move test files
    if create_directory('tests'):
        for test_file in ['test_predict.py']:
            if os.path.exists(test_file):
                move_file(test_file, 'tests')
    
    # 3. Remove System Junk
    # Remove __pycache__ directories
    cache_dirs_removed = remove_pycache_dirs()
    print(f"\nRemoved {cache_dirs_removed} __pycache__ directories")
    
    # Remove .pytest_cache directory
    if os.path.exists('.pytest_cache'):
        try:
            shutil.rmtree('.pytest_cache')
            print("Removed: .pytest_cache/")
        except Exception as e:
            print(f"Error removing .pytest_cache: {e}")
    
    # 4. Print final directory structure
    print("\nCleanup complete!")
    print_directory_tree()

if __name__ == "__main__":
    main()
