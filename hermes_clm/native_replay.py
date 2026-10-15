"""Original canonical envelopes offered to the host's durable replay verifier."""
from copy import deepcopy


class NativeReplayMixin:
    def get_context_replay_sources(self, messages):
        """Offer originals at the same coordinate and durable occurrence identity.

        This does not admit changed envelopes. Hermes must prove its supplied DB
        replay and the original committed row before reconstructing any losses
        in that replay. Normal full-envelope validation remains unchanged.
        """
        self._require_origin()
        self.store.validate_accepted()
        result = [None] * len(messages)
        if self._origin_mode != 'native':
            return result
        for index, message in enumerate(messages[:len(self.store.order)]):
            original = self.store.get(self.store.order[index]).envelope
            uid = message.get('message_uid')
            if type(uid) is str and uid and uid == original.get('message_uid'):
                result[index] = deepcopy(original)
        return result
