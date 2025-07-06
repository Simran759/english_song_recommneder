import shutil
import os

path = os.path.expanduser(r"C:\Users\Kanchan Infocom\AppData\Roaming\nltk_data")

try:
    shutil.rmtree(path)
    print("✅ Deleted nltk_data folder successfully.")
except FileNotFoundError:
    print("⚠️ Folder already deleted or not found.")
except PermissionError:
    print("❌ Permission denied. Try closing all Python apps and retrying.")
