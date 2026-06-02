import os
import sys

from app.indexer import load_index_json, search
from app.retrieval import query

load_index_json()
try:
    print(query("How long do refunds take?"))
except Exception as e:
    import traceback
    traceback.print_exc()
