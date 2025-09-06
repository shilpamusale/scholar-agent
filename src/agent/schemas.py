# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# src/agent/schemas.py

from pydantic import BaseModel, Field


class ToolInputSchema(BaseModel):
    """
    Defines the expected input schema for our agentic tools.
    Ensures that the LLM provides the full user question.
    """

    question: str = Field(
        description="The user's complete, original question that needs to be answered."
    )