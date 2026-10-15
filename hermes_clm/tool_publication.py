"""Retain an admitted edit until its native host publishes the matching reply."""
from .policies import PolicyViolation


class ToolPublicationMixin:
    def begin_tool_result_publication(self, name, call_id):
        """Opt a staged call into the host's canonical/persistence acknowledgment.

        Direct admission callers keep their existing immediate-commit contract.
        The host opts in before admission and resolves this same call afterward.
        """
        with self._tool_budget_lock:
            pending = self._pending_tool_budget
            if pending is None or (name, call_id) != (pending['name'], pending['call_id']):
                return False
            self._require_context_rollback_complete()
            pending['host_publication'] = True
            return True

    def resolve_tool_result_publication(self, name, call_id, *, published):
        """Release a confirmed publication, or retry its original rollback.

        Unknown database outcomes must remain owned by the host. Neither an
        exception nor a missing in-memory persistence marker proves absence.
        """
        if type(published) is not bool:
            raise ValueError('Publication resolution requires a known boolean outcome')
        with self._tool_budget_lock:
            pending = self._pending_tool_publication
            if pending is None or (name, call_id) != (pending['name'], pending['call_id']):
                return False
            if published:
                if pending.get('rollback_failed'):
                    raise PolicyViolation('tool_rollback_incomplete',
                        'An unpublished context edit must finish its original rollback')
            else:
                pending['rollback_failed'] = True
                self.telemetry.cancel_restore_publication(pending)
                self._restore_context_edit(pending['snapshot'], pending.get('publication_files'))
            # Physical recovery can raise: keep this service-owned preimage until
            # every file is restored. Nested host cleanup can safely retry it.
            self._pending_tool_publication = None
            return True
