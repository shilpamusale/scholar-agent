# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ingest.py: A command-line entrypoint for the RAG data ingestion pipeline.

This script serves as a simple, runnable interface to the core data processing
logic defined in `src.data_processing.ingestion`. Its primary responsibility is
to correctly configure the Python path to allow for the import of modules from
the `src` directory, and then to invoke the main ingestion pipeline function.

This allows a developer to manually trigger the creation or updating of the
ChromaDB vector store from the command line.
"""

# scripts/ingest.py

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_processing.ingestion import run_ingestion_pipeline  # noqa: E402

if __name__ == "__main__":
    run_ingestion_pipeline()
