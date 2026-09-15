import warnings
import traceback

# Store original showwarning
original_showwarning = warnings.showwarning

def patched_showwarning(message, category, filename, lineno, file=None, line=None):
    if "AttentionMaskConverter" in str(message):
        print("=== WARNING TRIGGERED ===")
        print(f"File: {filename}:{lineno}")
        print(f"Message: {message}")
        print("=== Call stack ===")
        traceback.print_stack()
        print("=== END ===")
    original_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = patched_showwarning

# Now import and run whatever the user runs
import transformers
print(f"Transformers version: {transformers.__version__}")
