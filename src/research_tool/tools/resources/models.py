"""BioModels: whether a curated model exists, and its main files (SBML) to download.

From its API spec (biomodels.org/docs/biomodels-jummp-swagger.json): GET /model/files/{modelId}?format=json
answers {"main": [{"name", "fileSize"}], "additional": [...]} (404 for an unknown id), and GET
/model/download/{modelId}?filename=<name> serves one file. Ids are BIOMD... (curated) or MODEL... .
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..http import HttpError, PoliteHttp

BIOMODELS_API = "https://www.ebi.ac.uk/biomodels"
_MODEL_ID = re.compile(r"^(BIOMD|MODEL)\d{10}(\.\d+)?$")


def biomodels_files(http: PoliteHttp, model_id: str) -> list[dict[str, Any]]:
    """The model's main files ({"name", "fileSize"}); HttpError for an unknown or malformed id."""
    if not _MODEL_ID.match(model_id.strip()):
        raise HttpError(f"not a BioModels id: {model_id!r} (expected BIOMD or MODEL and 10 digits)", status=404)
    files = http.get_json(f"{BIOMODELS_API}/model/files/{model_id.strip()}", params={"format": "json"})
    return [f for f in (files.get("main") or []) if isinstance(f, dict) and f.get("name")]


def download_url(model_id: str, name: str) -> str:
    return f"{BIOMODELS_API}/model/download/{model_id.strip()}?filename={name}"


def sbml_problem(path: Path) -> str:
    """'' if the file is SBML with a model in it, else what is wrong. A well-formedness check, not libSBML's."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return f"{path.name} is not readable XML: {exc}"
    if not root.tag.endswith("sbml"):
        return f"{path.name} is XML but not SBML (its root is <{root.tag}>)"
    if not any(child.tag.endswith("model") for child in root):
        return f"{path.name} is SBML without a <model>"
    return ""
