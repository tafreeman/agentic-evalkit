"""Persistence helper for tenant-scoped invoices.

Public contract:
- Every read and write is scoped to the supplied tenant.
- ``list_invoices`` excludes archived invoices unless ``include_archived`` is true.
- ``delete_invoice`` returns true only when one matching row was deleted.
"""

from sqlite3 import Connection
from typing import Any


class TenantStore:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def get_invoice(self, tenant_id: str, invoice_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT id, tenant_id, amount, archived FROM invoices WHERE id = ?",
            (invoice_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "tenant_id": row[1], "amount": row[2], "archived": bool(row[3])}

    def list_invoices(self, tenant_id: str, *, include_archived: bool = False) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT id, amount, archived FROM invoices WHERE tenant_id = ? ORDER BY id",
            (tenant_id,),
        ).fetchall()
        return [
            {"id": row[0], "amount": row[1], "archived": bool(row[2])}
            for row in rows
        ]

    def delete_invoice(self, tenant_id: str, invoice_id: str) -> bool:
        self._connection.execute(
            "DELETE FROM invoices WHERE tenant_id = ? AND id = ?",
            (tenant_id, invoice_id),
        )
        self._connection.commit()
        return True
