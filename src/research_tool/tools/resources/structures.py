"""Proteins and structures: UniProt sequences, AlphaFold DB predictions, PDB entries.

From their documentation (read 2026-11-21):
- UniProt (uniprot.org/help/api_retrieve_entries): rest.uniprot.org/uniprotkb/<accession>.fasta, with 404
  for an unknown accession and 410 for a removed one.
- AlphaFold DB (alphafold.ebi.ac.uk/api-docs): GET /prediction/<UniProt accession> lists the models, each with
  entryId, cifUrl, pdbUrl and globalMetricValue (mean pLDDT). The /api prefix is from memory; the live probe
  checks it.
- RCSB PDB (rcsb.org/docs/programmatic-access/file-download-services): files.rcsb.org/download/<id>.cif,
  and files.rcsb.org/header/<id>.cif for the header only.
"""

from __future__ import annotations

import re
from typing import Any

from ..http import HttpError, PoliteHttp

UNIPROT = "https://rest.uniprot.org/uniprotkb"
ALPHAFOLD = "https://alphafold.ebi.ac.uk/api"
RCSB_FILES = "https://files.rcsb.org"
# UniProt's documented accession pattern
_ACCESSION = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")
_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def accession(identifier: str) -> str:
    value = identifier.strip().upper()
    if not _ACCESSION.match(value):
        raise HttpError(f"not a UniProt accession: {identifier!r}", status=404)
    return value


def pdb_id(identifier: str) -> str:
    value = identifier.strip().lower()
    if not _PDB_ID.match(value):
        raise HttpError(f"not a PDB id: {identifier!r} (four characters, starting with a digit)", status=404)
    return value


def uniprot_fasta(http: PoliteHttp, identifier: str) -> str:
    text = http.get_text(f"{UNIPROT}/{accession(identifier)}.fasta")
    if not text.startswith(">"):
        raise HttpError(f"UniProt did not return a FASTA record for {identifier}", status=404)
    return text


def alphafold_models(http: PoliteHttp, identifier: str) -> list[dict[str, Any]]:
    models = http.get_json(f"{ALPHAFOLD}/prediction/{accession(identifier)}")
    found = (
        [m for m in models if isinstance(m, dict) and (m.get("cifUrl") or m.get("pdbUrl"))]
        if isinstance(models, list)
        else []
    )
    if not found:
        raise HttpError(f"AlphaFold DB has no model for {identifier}", status=404)
    return found


def pdb_header(http: PoliteHttp, identifier: str) -> str:
    return http.get_text(f"{RCSB_FILES}/header/{pdb_id(identifier)}.cif")
