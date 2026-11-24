"""Hand edits of the protocol: the researcher sets the effect size, materials and blinding; the rest follows.

The sample size, the allocation table, the data format, the checklist and the Markdown write-up are worked out
again from them (the same code the protocol agent uses). The design itself (conditions, outcome, tests, unit)
is the experiment plan's: to change it, request changes at the checkpoint with a note.
"""

from __future__ import annotations

from typing import Any

from ...core.editing import EditContext, EditResult, artifact_editor
from ...core.errors import ActionError, ArtifactError
from ...schemas.plan import ExperimentPlan
from .protocol import ProtocolConfig, make_protocol, render_protocol
from .schemas import Protocol

EDITABLE = ("effect_size", "materials", "blinding")


@artifact_editor("protocol")
def edit_protocol(value: dict[str, Any], ctx: EditContext) -> EditResult:
    previous = Protocol.model_validate(ctx.previous.payload)
    wanted = Protocol.model_validate({**previous.model_dump(), **{k: value[k] for k in EDITABLE if k in value}})
    if not 0 < wanted.effect_size <= 5:
        raise ActionError("the effect size (Cohen's d) must be above 0 and at most 5")
    materials = [m.strip() for m in wanted.materials if m.strip()]
    try:
        plan = ExperimentPlan.model_validate(ctx.store.envelope("experiment_plan").payload)
    except ArtifactError:
        raise ActionError("there is no experiment plan to compute the protocol from") from None
    seed = ctx.config.allocation_seed if isinstance(ctx.config, ProtocolConfig) else ProtocolConfig().allocation_seed
    protocol, allocation_csv = make_protocol(plan, wanted.effect_size, materials, wanted.blinding.strip(), seed)
    protocol.allocation_path, protocol.markdown_path = previous.allocation_path, previous.markdown_path
    files = {previous.allocation_path.rsplit("/", 1)[-1]: allocation_csv}
    if previous.markdown_path:
        files[previous.markdown_path.rsplit("/", 1)[-1]] = render_protocol(plan, protocol)
    return EditResult(protocol, files=files)
