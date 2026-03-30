"""Data lineage tracking.

Records the full transformation history of every clinical record so that
any output value can be traced back to its original source.  This is
critical for HIPAA compliance (proving *what* happened to data) and for
debugging pipeline issues.

Each transformation is represented as a ``LineageNode`` in a directed
acyclic graph.  The graph can be traversed forwards (source -> output)
or backwards (output -> source) and exported for compliance reports.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from healthpipe.exceptions import LineageError


class LineageNode(BaseModel):
    """A single node in the data lineage graph.

    Attributes:
        node_id: Unique identifier for this lineage node.
        record_id: The clinical record this node describes.
        operation: Name of the transformation (e.g. ``"ingest"``,
            ``"deidentify.ner"``, ``"date_shift"``).
        parent_ids: IDs of the upstream lineage nodes.
        timestamp: When the transformation occurred.
        metadata: Arbitrary key-value metadata about the transformation.
        checksum_before: Data checksum before transformation.
        checksum_after: Data checksum after transformation.
    """

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str = ""
    operation: str = ""
    parent_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksum_before: str = ""
    checksum_after: str = ""


class LineageTracker:
    """Tracks the full transformation lineage of clinical records.

    Usage::

        tracker = LineageTracker()
        node = tracker.record_operation(
            record_id="patient-123",
            operation="deidentify.ner",
            metadata={"entities_found": 3},
        )
        # later...
        history = tracker.get_history("patient-123")
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._by_record: dict[str, list[str]] = {}

    def record_operation(
        self,
        record_id: str,
        operation: str,
        parent_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        checksum_before: str = "",
        checksum_after: str = "",
    ) -> LineageNode:
        """Record a transformation and return the new lineage node.

        Args:
            record_id: Which clinical record was affected.
            operation: Description of the transformation.
            parent_ids: Upstream node IDs (auto-linked to last node if omitted).
            metadata: Additional details about the transformation.
            checksum_before: Data checksum before transformation.
            checksum_after: Data checksum after transformation.

        Returns:
            The newly created ``LineageNode``.
        """
        if parent_ids is None:
            existing = self._by_record.get(record_id, [])
            parent_ids = [existing[-1]] if existing else []

        node = LineageNode(
            record_id=record_id,
            operation=operation,
            parent_ids=parent_ids,
            metadata=metadata or {},
            checksum_before=checksum_before,
            checksum_after=checksum_after,
        )

        self._nodes[node.node_id] = node
        self._by_record.setdefault(record_id, []).append(node.node_id)
        return node

    def get_history(self, record_id: str) -> list[LineageNode]:
        """Return all lineage nodes for *record_id* in chronological order."""
        node_ids = self._by_record.get(record_id, [])
        return [self._nodes[nid] for nid in node_ids]

    def get_node(self, node_id: str) -> LineageNode:
        """Retrieve a specific lineage node by ID.

        Raises:
            LineageError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise LineageError(f"Lineage node not found: {node_id}")
        return self._nodes[node_id]

    def get_parents(self, node_id: str) -> list[LineageNode]:
        """Return the parent nodes of *node_id*."""
        node = self.get_node(node_id)
        parents: list[LineageNode] = []
        for pid in node.parent_ids:
            if pid in self._nodes:
                parents.append(self._nodes[pid])
        return parents

    def trace_to_source(self, node_id: str) -> list[LineageNode]:
        """Walk backwards from *node_id* to the original source.

        Returns:
            Ordered list from source to the given node.
        """
        path: list[LineageNode] = []
        visited: set[str] = set()
        stack = [node_id]

        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            node = self.get_node(current_id)
            path.append(node)
            stack.extend(node.parent_ids)

        path.reverse()
        return path

    @property
    def all_records(self) -> list[str]:
        """Return all record IDs that have lineage entries."""
        return list(self._by_record.keys())

    def to_dict(self) -> dict[str, Any]:
        """Export the full lineage graph as a serialisable dict."""
        return {
            "nodes": {
                nid: node.model_dump(mode="json") for nid, node in self._nodes.items()
            },
            "records": {rid: nids for rid, nids in self._by_record.items()},
        }
