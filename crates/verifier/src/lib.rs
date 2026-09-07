//! # Agent IR verifier
//!
//! The two rejection points of §4, implemented as two passes over the module.
//!
//! * [`Verifier::verify`] is the static phase of §14: is the program well
//!   formed, are its effect declarations legal, do its loops terminate, is
//!   every irreversible action guarded?
//! * [`Verifier::verify_safety`] is the second, distinct rejection point: does
//!   the agent actually *hold* the capabilities the program needs, and has a
//!   human approved the ones that are gated?
//!
//! The two are deliberately separate, because a program can be perfectly well
//! formed and still not be allowed to run.
//!
//! A rejected program is never executed. It comes back as [`Diagnostics`] —
//! which invariant broke, which operation broke it, and what to change — so
//! that the LLM builder gets a compiler error rather than a stack trace.
//!
//! ```
//! use agent_ir_verifier::Verifier;
//!
//! let module = agent_ir_parser::parse_module(r#"
//!     module @m version(0) {
//!       agent.func "f" {
//!         tool.call "delete_everything" {effect = #irreversible<db>}
//!         agent.return {effect = #pure}
//!       } {effect = #pure}
//!     }
//! "#).unwrap();
//!
//! let report = Verifier::new().verify(&module);
//! assert!(report.has_errors());
//! assert!(report.errors().any(|d| d.code == "I2"));
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use agent_ir_analysis::{dominates, enclosing_chain, EffectSummary};
use agent_ir_core::{
    Attribute, Diagnostic, Diagnostics, Effect, Module, OperationId, Type, ValueId,
};
use agent_ir_dialects::{AttrKind, Registry};

/// The default confidence below which an LLM-produced value must be verified
/// before it may feed an effectful operation (invariant I4).
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.8;

/// Checks a module against the invariants of §3.4 and the safety rules of §7.
#[derive(Clone, Debug)]
pub struct Verifier {
    registry: Registry,
    confidence_threshold: f64,
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Verifier {
    /// A verifier over the v0.1 dialects, at the default confidence threshold.
    pub fn new() -> Self {
        Verifier {
            registry: Registry::v0_1(),
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        }
    }

    /// Sets the I4 confidence threshold.
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// The dialect registry this verifier checks against.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// The static phase: well-formedness and invariants I1 to I5 (§14).
    ///
    /// Runs every check rather than stopping at the first failure, so the LLM
    /// builder sees the whole repair list in one round trip.
    pub fn verify(&self, module: &Module) -> Diagnostics {
        let effects = EffectSummary::build(module);
        let mut report = Diagnostics::new();

        for op in module.op_ids() {
            self.check_signature(module, op, &mut report);
            self.check_operands_dominate(module, op, &mut report);
            self.check_unknown_types(module, op, &effects, &mut report);
            self.check_irreversible_is_guarded(module, op, &mut report);
            self.check_llm_confidence(module, op, &effects, &mut report);
            self.check_loop_guard(module, op, &mut report);
            self.check_parallel_region(module, op, &effects, &mut report);
            self.check_terminators(module, op, &mut report);
        }

        self.check_top_level(module, &mut report);
        report
    }

    /// The safety phase: does the agent hold the capabilities the program needs
    /// (§2.3, §7)?
    ///
    /// This is where a `tool.call "delete_database"` without a matching
    /// capability is refused — at compile time, before any network call.
    pub fn verify_safety(&self, module: &Module) -> Diagnostics {
        let mut report = Diagnostics::new();

        for op in module.op_ids() {
            let operation = module.op(op);
            let effect = &operation.effect;
            if !effect.requires_capability() {
                continue;
            }

            let Some(capability) = module.capabilities.covering(effect) else {
                report.push(
                    Diagnostic::error(
                        "C1",
                        format!(
                            "`{}` declares {effect} but the agent holds no capability covering it",
                            operation.name
                        ),
                    )
                    .at(op)
                    .suggest(format!(
                        "grant a capability for `{}`{}, or lower the operation's effect if it \
                         really does not touch the world",
                        effect.class(),
                        effect
                            .scope()
                            .and_then(|s| s.name())
                            .map(|s| format!(" on scope `{s}`"))
                            .unwrap_or_default()
                    )),
                );
                continue;
            };

            if capability.requires_approval && !module.capabilities.is_approved(&capability.name) {
                report.push(
                    Diagnostic::error(
                        "C2",
                        format!(
                            "`{}` needs capability `{}`, which requires human approval that has \
                             not been recorded",
                            operation.name, capability.name
                        ),
                    )
                    .at(op)
                    .suggest(format!(
                        "obtain approval for `{}` before compiling, or route the action through \
                         HUMAN_REVIEW",
                        capability.name
                    )),
                );
            }
        }

        report
    }

    /// Both phases, in pipeline order. The safety phase only runs on a program
    /// that is already well formed, matching the pipeline of §4.
    pub fn verify_all(&self, module: &Module) -> Diagnostics {
        let mut report = self.verify(module);
        if !report.has_errors() {
            report.extend(self.verify_safety(module));
        }
        report
    }

    // ------------------------------------------------------------- structure

    fn check_signature(&self, module: &Module, op: OperationId, report: &mut Diagnostics) {
        let operation = module.op(op);
        let Some(signature) = self.registry.get(&operation.name) else {
            report.push(
                Diagnostic::error(
                    "S1",
                    format!("`{}` is not an operation of any v0.1 dialect", operation.name),
                )
                .at(op)
                .suggest("check the dialect list in §3.2, or register a new dialect"),
            );
            return;
        };

        if !signature.operands.accepts(operation.operands.len()) {
            report.push(
                Diagnostic::error(
                    "S2",
                    format!(
                        "`{}` takes {} operand(s) but was given {}",
                        operation.name,
                        signature.operands.describe(),
                        operation.operands.len()
                    ),
                )
                .at(op),
            );
        }
        if !signature.results.accepts(operation.results.len()) {
            report.push(
                Diagnostic::error(
                    "S2",
                    format!(
                        "`{}` produces {} result(s) but declares {}",
                        operation.name,
                        signature.results.describe(),
                        operation.results.len()
                    ),
                )
                .at(op),
            );
        }
        if !signature.regions.accepts(operation.regions.len()) {
            report.push(
                Diagnostic::error(
                    "S2",
                    format!(
                        "`{}` owns {} region(s) but has {}",
                        operation.name,
                        signature.regions.describe(),
                        operation.regions.len()
                    ),
                )
                .at(op),
            );
        }

        if signature.requires_literal && operation.literal.is_none() {
            report.push(
                Diagnostic::error(
                    "S4",
                    format!("`{}` needs a string literal naming what it does", operation.name),
                )
                .at(op)
                .suggest(format!("write `{} \"...\"`", operation.name)),
            );
        }

        for (key, kind) in signature.required_attrs {
            match operation.attr(key) {
                None => report.push(
                    Diagnostic::error(
                        "S3",
                        format!("`{}` requires the `{key}` attribute", operation.name),
                    )
                    .at(op),
                ),
                Some(value) if !matches_kind(value, *kind) => report.push(
                    Diagnostic::error(
                        "S3",
                        format!(
                            "`{}`'s `{key}` attribute should be {kind:?}, found `{value}`",
                            operation.name
                        ),
                    )
                    .at(op),
                ),
                Some(_) => {}
            }
        }

        // T2 of §14: the effect declaration must be one this operation is
        // allowed to make. An empty list means the operation may declare any.
        if !signature.allowed_effects.is_empty()
            && !signature.allowed_effects.contains(&operation.effect.class())
        {
            let allowed: Vec<String> = signature
                .allowed_effects
                .iter()
                .map(ToString::to_string)
                .collect();
            report.push(
                Diagnostic::error(
                    "T2",
                    format!(
                        "`{}` may not declare {}; it is limited to {}",
                        operation.name,
                        operation.effect,
                        allowed.join(", ")
                    ),
                )
                .at(op)
                .suggest(
                    "an operation that really has this effect belongs in a dialect that allows \
                     it — relabelling is what the effect system exists to prevent",
                ),
            );
        }
    }

    fn check_top_level(&self, module: &Module, report: &mut Diagnostics) {
        for op in module.top_level() {
            let operation = module.op(op);
            if !operation.name.is("agent", "func") {
                report.push(
                    Diagnostic::error(
                        "S1",
                        format!(
                            "`{}` cannot sit at module level; only `agent.func` can",
                            operation.name
                        ),
                    )
                    .at(op),
                );
            }
        }
    }

    fn check_terminators(&self, module: &Module, op: OperationId, report: &mut Diagnostics) {
        let operation = module.op(op);
        for &region in &operation.regions {
            for &block in &module.region(region).blocks {
                let ops: Vec<OperationId> = module
                    .block(block)
                    .ops
                    .iter()
                    .copied()
                    .filter(|&id| !module.op(id).erased)
                    .collect();

                for (position, &inner) in ops.iter().enumerate() {
                    let is_last = position + 1 == ops.len();
                    let terminator = self
                        .registry
                        .get(&module.op(inner).name)
                        .is_some_and(|s| s.terminator);
                    if terminator && !is_last {
                        report.push(
                            Diagnostic::error(
                                "S5",
                                format!(
                                    "`{}` ends its block, so nothing may follow it",
                                    module.op(inner).name
                                ),
                            )
                            .at(inner),
                        );
                    }
                }

                // A region that produces values has to say which ones.
                if !operation.results.is_empty() && !operation.name.is("agent", "func") {
                    match ops.last() {
                        Some(&last) if module.op(last).name.is("control", "yield") => {
                            let yielded = module.op(last).operands.len();
                            if yielded != operation.results.len() {
                                report.push(
                                    Diagnostic::error(
                                        "S5",
                                        format!(
                                            "`{}` declares {} result(s) but its region yields {}",
                                            operation.name,
                                            operation.results.len(),
                                            yielded
                                        ),
                                    )
                                    .at(last),
                                );
                            }
                        }
                        _ => report.push(
                            Diagnostic::error(
                                "S5",
                                format!(
                                    "`{}` declares results, so its region must end with \
                                     `control.yield`",
                                    operation.name
                                ),
                            )
                            .at(op),
                        ),
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- I1, T1

    fn check_operands_dominate(&self, module: &Module, op: OperationId, report: &mut Diagnostics) {
        for (index, &operand) in module.op(op).operands.iter().enumerate() {
            if !dominates(module, operand, op) {
                report.push(
                    Diagnostic::error(
                        "I1",
                        format!(
                            "operand {index} of `{}` has no dominating producer",
                            module.op(op).name
                        ),
                    )
                    .at(op)
                    .suggest(
                        "move the producing operation before this one, or hoist it out of the \
                         region that hides it",
                    ),
                );
            }
        }
    }

    fn check_unknown_types(
        &self,
        module: &Module,
        op: OperationId,
        effects: &EffectSummary,
        report: &mut Diagnostics,
    ) {
        // The summary, not the declaration: a `control.loop` is declared
        // `#pure` and may still hand the value to a tool forty times.
        if effects.is_pure(op) {
            return;
        }
        let operation = module.op(op);
        for &operand in &operation.operands {
            if module.value(operand).ty == Type::Unknown {
                report.push(
                    Diagnostic::error(
                        "I1",
                        format!(
                            "`{}` performs {} but consumes a value of unresolved type; an \
                             `Unknown` may not cross an effect boundary",
                            operation.name,
                            describe_effects(effects.of(op))
                        ),
                    )
                    .at(op)
                    .suggest("resolve the value's type, or insert a `core.cast` that does"),
                );
            }
        }
    }

    // -------------------------------------------------------------------- I2

    fn check_irreversible_is_guarded(
        &self,
        module: &Module,
        op: OperationId,
        report: &mut Diagnostics,
    ) {
        let operation = module.op(op);
        if !operation.effect.is_irreversible() {
            return;
        }

        let guarded = preceding_ops(module, op).into_iter().any(|earlier| {
            let candidate = module.op(earlier);
            if !candidate.name.is("agent", "verify") {
                return false;
            }
            match candidate.str_attr("capability") {
                Some(name) => module.capabilities.authorizes(name, &operation.effect),
                None => false,
            }
        });

        if !guarded {
            report.push(
                Diagnostic::error(
                    "I2",
                    format!(
                        "`{}` declares {} without a preceding `agent.verify` naming a capability \
                         that covers it",
                        operation.name, operation.effect
                    ),
                )
                .at(op)
                .suggest(
                    "insert `agent.verify {capability = \"...\"}` before this operation, or route \
                     it through human approval",
                ),
            );
        }
    }

    // -------------------------------------------------------------------- I3

    fn check_parallel_region(
        &self,
        module: &Module,
        op: OperationId,
        effects: &EffectSummary,
        report: &mut Diagnostics,
    ) {
        if !module.op(op).name.is("control", "parallel") {
            return;
        }
        for &region in &module.op(op).regions {
            let siblings = module.ops_in_region(region);
            for (i, &left) in siblings.iter().enumerate() {
                for &right in &siblings[i + 1..] {
                    if effects.conflict(left, right) {
                        report.push(
                            Diagnostic::error(
                                "I3",
                                format!(
                                    "`{}` and `{}` conflict on an overlapping scope and cannot \
                                     share a `control.parallel` region",
                                    module.op(left).name,
                                    module.op(right).name
                                ),
                            )
                            .at(right)
                            .suggest(
                                "sequence the two conflicting operations, or narrow their effect \
                                 scopes if they really touch different resources",
                            ),
                        );
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------- I4

    fn check_llm_confidence(
        &self,
        module: &Module,
        op: OperationId,
        effects: &EffectSummary,
        report: &mut Diagnostics,
    ) {
        // As with unresolved types, the question is what the operation really
        // does, not what its own line declares.
        if effects.is_pure(op) {
            return;
        }
        let operation = module.op(op);

        let verified: Vec<ValueId> = preceding_ops(module, op)
            .into_iter()
            .filter(|&earlier| module.op(earlier).name.is("agent", "verify"))
            .flat_map(|earlier| module.op(earlier).operands.clone())
            .collect();

        for &operand in &operation.operands {
            let value = module.value(operand);
            if value.provenance.needs_verification(self.confidence_threshold)
                && !verified.contains(&operand)
            {
                report.push(
                    Diagnostic::error(
                        "I4",
                        format!(
                            "`{}` performs {} but consumes an LLM value with confidence {:.2}, \
                             below the {:.2} threshold, and nothing verified it first",
                            operation.name,
                            describe_effects(effects.of(op)),
                            value.provenance.confidence,
                            self.confidence_threshold
                        ),
                    )
                    .at(op)
                    .suggest(
                        "add an `agent.verify` on that value before this operation, or obtain it \
                         from a tool rather than from the model",
                    ),
                );
            }
        }
    }

    // -------------------------------------------------------------------- I5

    fn check_loop_guard(&self, module: &Module, op: OperationId, report: &mut Diagnostics) {
        let operation = module.op(op);
        if !(operation.name.is("control", "loop") || operation.name.is("control", "while")) {
            return;
        }
        match operation.int_attr("max_iterations") {
            Some(bound) if bound > 0 => {}
            Some(bound) => report.push(
                Diagnostic::error(
                    "I5",
                    format!("`{}` declares max_iterations = {bound}", operation.name),
                )
                .at(op)
                .suggest("a termination guard must be a positive number of iterations"),
            ),
            None => report.push(
                Diagnostic::error(
                    "I5",
                    format!("`{}` has no termination guard", operation.name),
                )
                .at(op)
                .suggest(
                    "add `max_iterations = N`, or prove progress another way and record it as an \
                     attribute",
                ),
            ),
        }
    }
}

/// Every operation that certainly runs before `op`.
///
/// That is: earlier operations in the same block, and earlier operations in
/// every enclosing block. Operations in a sibling region are excluded, because
/// nothing says they ran.
fn preceding_ops(module: &Module, op: OperationId) -> Vec<OperationId> {
    let mut preceding = Vec::new();
    for frame in enclosing_chain(module, op) {
        for &earlier in &module.block(frame.block).ops[..frame.position] {
            if !module.op(earlier).erased {
                preceding.push(earlier);
            }
        }
    }
    preceding
}

fn matches_kind(value: &Attribute, kind: AttrKind) -> bool {
    match kind {
        AttrKind::Any => true,
        AttrKind::Int => matches!(value, Attribute::Int(_)),
        AttrKind::Float => matches!(value, Attribute::Float(_) | Attribute::Int(_)),
        AttrKind::Bool => matches!(value, Attribute::Bool(_)),
        AttrKind::Str => matches!(value, Attribute::Str(_)),
        AttrKind::Dict => matches!(value, Attribute::Dict(_)),
    }
}

/// Formats a set of effects for a diagnostic.
fn describe_effects(effects: &[Effect]) -> String {
    let mut names: Vec<String> = effects
        .iter()
        .filter(|e| !e.is_pure())
        .map(ToString::to_string)
        .collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        "#pure".to_string()
    } else {
        names.join(" + ")
    }
}
