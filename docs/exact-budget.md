# Exact request budgets

`CLMConfig.budget_mode` defaults to `approximate` for legacy callers. This mode
does not certify a rendered prompt budget. The isolated qualification arm uses
`budget_mode="exact"`, `active_target_tokens=32768`, `reserve_tokens=4096`,
`output_tokens=4096`, and `safety_margin_tokens=1024`.

Exact mode requires an explicitly bound count service:

```python
backend = LlamaServerTokenizer(identity, fresh_host_attestation, transport)
counter = ExactRequestCounter(backend, parity_certificate)
engine.bind_exact_counter(counter, identity)
```

`transport` is `HttpCounterTransport` pointed at the same provider, with the
owning profile's credentials supplied explicitly. It can call only `/props`,
`/apply-template`, `/tokenize`, and `/v1/chat/completions/input_tokens`; it cannot
perform inference. No count request is made during construction or binding.

The host attestation callback must inspect the current running process and its
model, tokenizer, executable/build, preprocessing configuration and actual
context ceiling. A constant callback returning a historical identity is not a
live attestation. Runtime props independently verify model alias, template hash,
build and context ceiling. Fresh endpoint health and identity are required even
on a cache hit. A restart or changed identity requires requalification/rebinding.

`ParityCertificate.from_observations` validates at least 30 distinct complete
payloads covering tools, parallel calls/results, resumed history, null and block
text, Unicode, code, reasoning and template options. It requires retained token
IDs and provider prompt usage with one fixed BOS difference of at most one token.
It does not collect evidence. `offline_fixture` evidence always produces
unqualified counts. `live_provider` must refer to independently collected actual
provider receipts; setting that label is not itself evidence. Live qualification
and permission to use the shared endpoint are separate gates.

The current adapter supports text-only chat completions. Media, route changes,
unattested auxiliary adapters/callbacks, per-request headers or query overrides,
client custom headers/query defaults, and SDK retries are unsupported in exact
mode. Use zero SDK retries so every explicit attempt reaches the final gate.
Unsupported paths stop locally instead of silently falling back to estimates.

Every dispatch counts the complete effective SDK body after middleware and
stream options, including `extra_body` overrides, tools, framing and template
settings. Its final messages must also contain a closed valid tool protocol;
late overrides cannot introduce orphaned, duplicate or incomplete groups.
The bound identity supplies the actual provider ceiling. All supplied
output-cap fields must preserve the configured allowance; a request cannot gain
prompt space by quietly reducing generation. The normal threshold is target
minus reserve. Restore/pending-group reasons can use the reserve; requests above
the operating target require a specific recovery reason and return-below-normal
plan. A pressure episode allows at most six attempts and two consecutive
nonprogress edits. Re-declaring a plan does not reset those bounds.

`clm_restore` accepts an optional `return_plan` for a temporary headroom restore.
Host-controlled management may call `begin_budget_excursion(reason, return_plan)`.
Neither route permits a hard provider-window violation or protected-source loss.

Edits stage a prospective payload using the last accepted complete task frame.
Native source projection and the host's pure transport conversion produce the
candidate messages; all other final fields remain intact. The current rendering
must match that accepted frame before any replacement. Unrepresented middleware
changes or a stale canonical branch reject with `candidate_frame_mismatch`.
There is no guessed inverse transform. Summary/auxiliary requests never replace
the task frame. Newly generated assistant/tool tails are independently counted
at the next actual dispatch. A native context-tool edit also checks the current
assistant and known sibling-result tail before returning its provisional reply.
The host supplies its canonical branch and real tool-call ID; a stale branch,
foreign ID or unresolved previous edit refuses before mutation.

For these edits, state files and success events remain unpublished until the
host prepares the actual tool result. Hermes calls the optional
`admit_tool_result` hook after result transformations, spilling, hints and
serialization, before appending the result to canonical history. CLM counts
the same known tool frame around both the old and proposed source views. This
enforces actual reply overhead, restoration growth and one pressure observation.
A rejected result restores the accepted source, revision, pressure and files,
and becomes a structured tool error. The companion `abort_tool_result` hook
discards an owned unpublished stage if reply preparation fails or exits early.
The sequential caller owns cleanup throughout dispatch, execution, spinner
completion and publication, including exceptions and interrupts before reply
preparation begins.
Publication keeps that ownership until state persistence and event publication
both return. Ordinary exceptions and interrupts restore the accepted runtime
and the owned session checkpoint/event files before returning an error or
propagating the interrupt. A failed rollback remains pending and fail closed;
outer cleanup can retry it. Checkpoint writes remove their own temporary file
on exceptional exit. This is in-process rollback, not cross-file crash recovery.
The CLM commit completes before Hermes appends the canonical tool reply and its
SessionDB row. An interrupt after that commit can leave the accepted context edit
in place without its later acknowledgment. Original envelopes remain retained;
cancellation at this point does not mean the pre-edit working context was restored.
This boundary is separate from rollback of an unpublished or failed CLM commit.
Native interruption/end/resume continuity at this boundary must be verified before
live delivery; the offline post-commit probe does not establish later recovery.
Owned exact edit commits require successful event publication when event logging
is enabled: append-open, write and close errors propagate into the same rollback,
including partial appends. Outside those owned commits, event logging retains its
existing best-effort convention; disabling event logging requires no event file.
Incomplete rollback revokes publication permission and blocks normal engine,
store and session operations until the owned abort succeeds. Independently
called engine edits also restore their owned checkpoint/event files on failed
publication; public engine and lower-store edits roll back on interruption
during candidate counting before propagating the interrupt.
If direct file restoration fails, the engine retains its preimage and blocks
ordinary operations until `retry_context_rollback` succeeds. Final-request
admission retries that recovery, but refuses the current request even after
success; the caller must rebuild from the accepted state and admit it again.
The task guard leaves renderer binding unchanged while direct recovery is
pending. Lifecycle callers that stop before transport can explicitly retry
with `retry_context_rollback` once the resource is available.
The host retains the engine that executed the tool. An attachment change with
an outstanding stage rolls that owner back and stops the turn; a completed
approximate edit has no stage and preserves its existing transition behavior.
Source/budget refusals during a tool round leave immediately, without generic
error-result synthesis or normal turn finalization.
Other context engines can leave both hooks unset.

Unfinished parallel siblings remain explicitly pending in candidate counts;
CLM never invents their results. Such counts authorize publishing only the
context edit. Every actual provider dispatch still requires a complete tool
group and independently counts the full effective body. A caller that omits
reply admission cannot dispatch its staged edit.

Protection, protocol and exact candidate checks precede atomic persistence.
Rejected edits preserve accepted state, sources, revision and success events.
Independent lower store writers use the same guard; a public store transaction
does not exempt them. Compound engine edits validate their final staged state.
An independent lower writer retains an engine undo callback through its entire
admission boundary. Failure after pressure observation or completed validation
restores pressure, excursion, accepted frame and last decision along with the
original store state; compound edits retain their existing outer engine owner.
Errors report a specific code and available/required tokens when known; count
outages report unknown required tokens. Optional restoration limits use rendered
prompt growth, not the legacy per-message estimate.

Status and edit telemetry distinguish the last final-budget decision from store
estimates. Direct edits label their source-frame scope; host-admitted edits label
the known tool-frame scope and whether its group is complete. The accepted
source frame remains separate from that tool-frame receipt. Unit or native-loop tests with injected counts prove control
flow and arithmetic only; they are not served-model parity or operational proof.
