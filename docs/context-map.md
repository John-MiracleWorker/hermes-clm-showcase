# Context Map source index

The experimental index is disabled by default. It provides retained-source
provenance, topology, checkpoint decoding, semantic authoring on existing
archive/compact operations, request-only navigation snapshots, lookup and
archive-reference restoration. Native continuity, served-counter qualification
and practical agent-selected use have separate acceptance gates. Offline checks
alone do not establish complete delivery.

`ContextStore.context_map_index` returns the shared `ContextMapIndex` facade.
`create_node(path, source_uids, description=..., keywords=...)` accepts a complete
verified source span and assigns operational IDs. Normalized paths reuse nodes;
labels are untrusted data and grant no source-edit or tool permissions.
`attach_archive_ref` links an existing host reference, and `update_node` changes
bounded labels or attributed semantic links. Public writers use the existing
accepted store mutation guard, including exact-budget admission and rollback.
All public APIs independently validate canonical source and map integrity.
Reads return detached metadata and never restore or execute retained tools.

The root has depth zero and new semantic paths have at most five edges, subject
to the configured creation limit. Labels have hard limits of 96 title characters,
256 description characters and 16 keywords of 64 characters each. Existing
persisted topology is validated against the hard limits even when settings are
lowered or the feature is disabled; it is never discarded to fit a new setting.

When enabled, `clm_archive` and both forms of `clm_compact` accept optional
`map_path`, `map_description`, and `keywords`. Their detached schemas expose the
hard bounds; the tools require exact JSON lists and strings without coercion.
A supplied proposal while authoring is disabled refuses with
`context_map_disabled`; description or keywords without a path refuse with
`map_path_required`. Invalid shapes, bounds, or newly created ancestry refuse
with `invalid_map_metadata`. Omitted fields preserve ordinary operation behavior.
Empty or omitted description/keywords preserve existing authored metadata, and
normalized existing paths remain reusable under lowered creation limits.

Semantic placement shares the source operation's transaction. The host supplies
complete raw or derived source closure and allocates the node/reference IDs.
Only presentations introduced by this operation move from their initial
automatic placement to the semantic node. Earlier shared associations, ordering,
body history, semantic links, and fallback raw references remain intact. Reused
presentations gain the semantic association without losing earlier ones.

Archive/compact and manual/emergency compression events retain their actual
`model`, `host-manual`, or `host-emergency` initiator. Their `context_map_delta`
records actual before/after map revisions, changed nodes, source-associated raw
references, introduced presentation associations, and verified source closure.
A stable raw reference may be reported again when reused; map revisions need not
advance exactly once per operation. Labels cannot choose attribution. Mapped
source operations require successful physical event publication when logging is
enabled, including heuristic mode, disabled checkpoint persistence, and retained
maps whose new authoring is disabled. A partial event write rolls back through
the existing transaction/file owner; logging disabled remains an explicit choice.

Map-bearing checkpoint version 6 explicitly records native versus standalone
coordinate origin. The index, stable reference allocation, source-group witnesses,
summary/body presentation provenance and map revision live inside the same state
object as canonical sources. Existing strict version 4/5 formats remain valid.
Temporary candidate decoding validates provenance and topology before adoption.
Current groups and retained derivation witnesses permit deterministic backfill.
Historic tool joins without proof require recovery with the original checkpoint
preserved. Current references survive append, archive, restore and restart;
rewound references remain auditable and reject current-branch use.

Verified legacy reconstruction retains a transient pending observation. Pure
decoding does not publish an event or alter telemetry metrics. Admitted session
startup or a directly used store's `save()` consumes it into one
`context_map_backfill` event attributed to `host-backfill`; historical attribution
is explicitly `unknown`. The event is recorded before checkpoint construction,
so persisted metrics agree. Save within an existing transaction queues into that
owner's buffer. A direct standalone save owns the state/event preimages itself.
Publication failure restores the pending observation and owned files for retry.
If restoration itself fails, the store retains the original runtime, counter and
file preimages and refuses ordinary operations with `context_rollback_incomplete`.
Call `store.retry_save_rollback()` to retry that same restoration. Ownership stays
held during each attempt and is released only after success; then a later save
publishes the pending observation once. The retry returns `True` after recovery
and `False` when no recovery is pending. This recovery owner is in-process state.
Logging disabled consumes the observation without a physical event, independently
of checkpoint persistence. Temporary export validation, recovery-required state,
repeated successful saves, and version-6 reloads do not publish a new backfill.

The map's scope records its creation session as a stable identifier anchor.
An admitted successor session of the same canonical conversation preserves that
anchor and its references while updating the store's current session telemetry.
Conversation and native/standalone origin must still match; node links and raw
references stay within that retained lineage.

Witnesses record complete host-observed groups before rewind, with ordered source
UIDs, original-envelope digests, positions and the observed branch digest. They
are retained host provenance, not authentication of arbitrary externally forged
checkpoint history. Import through the public facade cannot mint witnesses,
references or operational IDs. Reference use still checks the actual current
complete canonical group; labels and saved metadata cannot grant restoration of
a different branch.

Host revision, creation/update timestamps and access facts are authoritative at
public import. Authored labels and valid existing-node topology may be proposed;
host metadata changes refuse atomically. Newly proposed ancestry paths obey the
current configured depth, including affected descendants; changed descriptions
obey the current description limit. Unchanged saved deeper paths and longer
descriptions remain valid within the persisted hard limits when settings are
lowered. Meaningful node changes update the host
modification time and map revision; no-op writes and reads do not churn them.
Stored map records require exact plain JSON and UTF-8 strings. Create/update
keyword tuples are normalized to JSON lists before adoption.

Direct diagnostic ingestion retains shape-valid invalid/pending observations.
Their separate `diagnostics` registry records the actual observed UID branch,
original-envelope digests and independently rechecked uncertified sources.
Reachable node `diagnostic_refs` supply audit coverage without an operational
archive reference or restore permission. Exact source bytes remain in the store.
A valid subsequent authoritative replay can recover and persist these audit
records; invalid current protocol still refuses accepted operations. Arbitrary
unproved legacy ghosts do not gain audit coverage, and synthetic derivations
still require proven complete source groups. A later actually observed complete
group may independently establish normal archive authority for its sources.

`ContextMapIndex.render_snapshot()` supplies a detached off-prompt structural
overview. The engine's separate request renderer appends one `CLM ARCHIVE MAP`
JSON block to an existing selected user-content carrier when enabled and the
index contains retained references or diagnostics. It preserves the carrier's
original content, all system/developer messages, role order and canonical
sources. Host IDs/counts and explicitly untrusted authored labels are separate
fields; labels are escaped data. Raw source and summary bodies are not map text.

Root, a bounded top-level overview, and deterministic relevant/recent paths are
candidates. Relevance uses lexical overlap with the most recent canonical user
text; recency uses stored access/update values and stable ID ties. Omission
counts retain awareness of the complete off-prompt index. Rendering does not
advance revisions, timestamps or access counters. Small maps are not padded.

Selection before a complete provider request is available is provisional.
Hermes calls the optional generic `prepare_context_request` hook at its concrete
chat dispatch boundary before the separate read-only validator. It supplies the
actual route and pure provider message renderer. CLM proves the selected and
actual messages agree, then fits candidates using the complete request. The
hook can replace effective messages only: tools, model, template options,
output allowance and SDK `extra_body` precedence remain fixed. Unrepresented
transforms refuse. An unfinished rollback or tool publication stays with the
existing final-validator recovery/abort owner, which refuses that dispatch.

Exact `context_map.snapshot_tokens` is the marginal difference between two
complete counted requests with and without the framed block on the same carrier.
Telemetry includes both prompt counts/fingerprints and the
`complete_request_difference` scope. This includes tokenization at the content
boundary and all actual template/schema effects, not a character estimate or
standalone-message approximation. The target is 1,500 and hard cap 2,000 by
default; mandatory root/framing may exceed the target but never the cap. If the
minimum does not fit, dispatch refuses with `context_map_cap_exceeded`. Missing,
stale, approximate or unavailable exact counts fail closed. Every token also
remains in final `P`, subject to the existing S/T/H/O/M policy.

Standalone exact callers use `prepare_context_request(request, route=route)`
before `check_final_request(request, route=route)`, then send exactly that
prepared request only if allowed. The validator independently rechecks the cap
and total budget; validation never rewrites a request. A first resumed request
needs no previous budget frame. Prospective edits and prepared tool replies
measure their own complete frames with the same presentation. Approximate mode
retains explicitly approximate behavior and provides no exact-cap certification.
Offline deterministic counters verify control flow, not served tokenizer parity.

`search_nodes()` and enabled-only `clm_lookup` share deterministic local metadata
ranking. No retrieval-quality, autonomy or live provider qualification is
established by these offline slices.
An omitted `search_nodes` result limit uses the configured cap; explicit limits
must be exact integers within that cap.

Mutable map settings are validated at accepted store consumers and before direct
ingestion, including selection and persistence-disabled admission. Mode must be
an exact string with value `deterministic`; caller-defined equality cannot supply
the mode. In Hermes configuration loading, an explicit null map option remains
unset and resolves to its correctly typed default. Direct configuration and
mutated runtime null values refuse. Engine
observation, session start/end and projected-selection interruptions restore
ownership and canonical-ingestion flags before propagating the interruption;
retry may then proceed. This does not extend the separate postcommit canonical
acknowledgment guarantee. The normal package includes the retrieval subpackage.

Session start, turn completion and session end capture their owned checkpoint and event-file
preimages before ingestion or publication. An unpublished failed operation
restores those files as well as source/runtime state and original startup
services. Standalone startup captures the intended target before loading it,
including initially absent files. Prior/target startup restoration attempts every
affected engine even when one fails. Distinct files retain independent owners;
engines sharing the same checkpoint retain admission together so a delayed retry
cannot overwrite a newer admitted publication. If file restoration fails,
the existing rollback owner retains its preimage and blocks ordinary admission
until `retry_context_rollback()` completes the restoration.

## Ranked metadata lookup

`clm_lookup(query, max_results?, cursor?)` returns at most five validated
`(node_id, archive_ref)` candidates. The default is the configured result cap.
It searches semantic labels and paths; it never restores sources or returns raw
source/summary bodies. Titles and descriptions are untrusted navigation data.
Historic results explicitly have `current_branch: false`. A reference is not
permission to bypass current-branch or restoration budget checks.

Ranking version 1 uses NFKC/casefold and unique Unicode alphanumeric query terms.
Each term matches as a substring, scoring 8 for the title, 6 for keywords, 4 for
ancestor titles and 2 for the description. The two most recent distinct semantic
update instants add 2 and 1; access contributes at most 3. Zero lexical matches
remain excluded. Ties use host node IDs. These are fixed starting weights, not
measured optimal values. Pure index search and rendering do not record access.

Each result contains a bounded path, description and keyword sample, one complete
validated reference, source count/index range, up to four provenance records and
up to two summary IDs. Explicit omission counts preserve the distinction between
a bounded result and its complete retained provenance. Token estimates are
approximate retained-source counts, not exact restore requirements.

Stored node estimates refresh when their retained-source membership changes.
Usage calibration alone preserves those estimates, semantic modification times
and map revision. New or changed membership uses the current approximate counter
over original retained envelopes; exact request admission remains independent.

Use the returned cursor with the same query and limit. Pages finish an unfinished
node before advancing through other nodes. Each admitted page records one access
per distinct presented node; access can reorder visited nodes without skipping
unseen references. Multiple semantic aliases remain distinct candidates.
`rank` denotes the consecutive pagination sequence. The cursor binds accepted
searchable metadata, provenance, branch eligibility, access and configuration.
Ordinary canonical append, calibration feedback and restart preserve it; changed lookup state returns
`stale_lookup_cursor`, requiring a fresh search. Cursors do not grant permissions.

Lookup uses the existing transaction owner. Exact native calls publish access,
metrics and events only after the actual prepared reply passes full request
budget admission. Changed result metadata, blocked/error replies, unavailable
counts and publication failures reject and roll back. A failed file rollback
retains its owner until explicit recovery succeeds. Direct exact calls qualify
the prospective source frame; actual native reply admission is a separate check.
Lookup events record candidate presentation and monotonic operation duration,
not source consultation, recall accuracy or autonomous retrieval success. Those
unavailable measurements are null. Complete native continuity remains separately required work.

## Optional browse tool omission

The optional interactive navigation tool `clm_browse_map(node_id, cursor?)` described in the preliminary roadmap is intentionally omitted from the tool surface because pilot navigation evidence showed no requirement for interactive tree browsing. The configuration setting `context_map_browse: bool = False` remains strictly guarded in `CLMConfig.validate_context_map()` to refuse activation until separately verified navigation support is provided.

## Restoring lookup references

With Context Map enabled, pass one through five exact `archive_refs` returned by
lookup to `clm_restore`, optionally with a bounded `reason` and `return_plan`.
Use references separately from legacy message/summary selectors. Duplicate refs
resolve once, through current-branch validation and complete tool-group policy.
Unknown references return `unknown_archive_ref`; historic references return
`historic_branch_conflict`. Authored labels and reasons cannot change protection.

Restoration loads original envelopes from archives, summaries or body
compactions. The result reports complete `restored_ids`, actual
`changed_source_ids`, bounded host `source_node_ids` and an omission count.
Already active originals produce `noop: true` with zero source expansion and no
added source thrash or semantic revision. `tokens_restored` has explicit scope
`source_delta_estimate`. Exact complete-request counts remain separate and include
the actual native assistant/tool reply, schemas, template options and reserved
output. Temporary headroom requires a return plan and remains bounded by the
existing pressure episode, provider ceiling and restoration growth limit.

Native reference restoration stays staged until the actual prepared reply is
admitted. JSON values and types must match the owned result; formatting and object
key order may differ. A changed result returns `restore_result_transformed` and
rolls back. Error, blocked, cancelled, unavailable-count, oversized-frame and
publication-failure paths retain existing rollback ownership. A failed rollback
must finish restoring its original preimages before ordinary admission resumes.

The engine retains at most 32 recent admitted lookup pages in runtime. A restore
event associates each ref with its newest matching host receipt, including the
original query, normalized query, lookup ID, candidate node and rank. The tool
result returns bounded `lookup_ids` and `unattributed_archive_refs`; complete
queries remain in the off-prompt event. This association works with event logging
disabled. Cancelled lookup pages confer no provenance. Eviction, session reset,
restart or a fresh agent yields explicit unknown provenance instead of an
inferred query. These runtime receipts are not persisted in checkpoints.

Status exposes `reference_restore_observations`: bounded current-engine-instance
operational counters and 32 recent outcomes, explicitly marked nonpersistent.
Nested direct/tool calls count once. A staged native result is not a success;
actual admitted publication is. Failure and invalid-ref observations are separate
from transactional store metrics and files, preserving refusal atomicity even
when file publication fails. Recovery retries do not count the same failed stage
again. Each outer invocation owns only the transaction it creates. A separate
refused call records its own failure without consuming the pending call's outcome
or inheriting its lookup provenance, including while failed rollback awaits retry.
A plugin block before loader execution is outside this observer. Legacy
selectors retain their previous accounting. `restored_tokens` in these
observations is a source estimate; source consultation, wrong-restore evaluation
and autonomous success remain null.

Successful reference events measure the source operation from entry to
`apply_restore` through source and map changes, ending before event recording,
persistence and native prepared-reply admission. `started_monotonic_ns` and
`completed_monotonic_ns` define `restore_latency_ms`, with explicit scope
`source_operation_before_publication`. Publication and native admission latency
are unmeasured and null. The staged event retains that interval when published.
Events carry explicit UTC and America/Detroit timestamps for the same instant.

Each counted outcome receives its own `observed_monotonic_ns`, `timestamp_utc`
and `timestamp_detroit` at the actual success or failure boundary. Staging adds
no success observation. Outcome `whole_call_latency_ms` remains null because
these observations do not measure the whole call or host acknowledgment.

## Native result publication and replay

Prepared-reply admission retains a separate in-process publication owner until
Hermes appends the exact result to canonical history and acknowledges its normal
SessionDB flush. Without persistence, canonical append is the acknowledgment.
An interruption before that boundary restores the owned runtime, map and file
preimages. If the DB commit succeeded but its acknowledgment was interrupted,
the actual durable occurrence can prove publication; a later UI callback failure
does not undo that committed result. An unavailable or ambiguous durable outcome
retains ownership and refuses further dispatch or ordinary persistence until it
can be resolved. Recovery uses the original engine even if the agent's current
engine attachment changes.

A publication failure corrects only its own reference-restore success observation.
Independent outcomes and recent-history eviction remain intact. A failed physical
rollback keeps its original owner for retry through the native next-turn, session
end or close consumer. This does not create a joint crash-atomic transaction
between the CLM checkpoint and SessionDB.

Fresh native resume may recover fields that SessionDB's conversation projection
omits. The engine offers retained originals at matching coordinates and durable
message identities; Hermes verifies the unique current DB row, its recorded
payload snapshot and every projected field before reconstructing the original
envelope. Both supported projections, with and without row IDs, are checked.
Changed supplied fields, changed durable metadata, missing/ambiguous rows and
failed reads refuse before a provider send. The caller's history is detached
before native repair, and normal full-envelope validation remains in force.

Explicit native `AIAgent.close()` drains only its owned title producers before
tearing down the session. A producer still active after the ten-second drain
raises `TimeoutError`; the caller must retry close after it finishes. The pending
attempt preserves the session and ownership. Deferred, unstarted title work is
retired before a successful close. The existing bounded interpreter-exit drain
remains separate; this retry contract does not promise completion after abrupt
process termination or cancellation of arbitrary foreign callbacks.
