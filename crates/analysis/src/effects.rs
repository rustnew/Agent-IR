//! Effect summaries: what an operation does to the world, nested regions
//! included.
//!
//! An operation's own `effect` field describes only itself. A `control.loop`
//! is declared `#pure`, but running it may delete a database forty times. Every
//! §5 pass condition that talks about "the effect of this operation" means the
//! summary, not the declaration, and reading the declaration instead is the
//! single easiest way to build an unsound pass.

use agent_ir_core::{Effect, Module, OperationId};
use std::collections::HashMap;

/// The effects each operation performs, transitively.
#[derive(Clone, Debug, Default)]
pub struct EffectSummary {
    summaries: HashMap<OperationId, Vec<Effect>>,
}

impl EffectSummary {
    /// Computes a summary for every live operation in the module.
    pub fn build(module: &Module) -> Self {
        let mut summary = EffectSummary::default();
        for op in module.top_level() {
            summary.visit(module, op);
        }
        summary
    }

    fn visit(&mut self, module: &Module, op: OperationId) -> Vec<Effect> {
        let operation = module.op(op);
        let mut effects = vec![operation.effect.clone()];
        let regions = operation.regions.clone();
        for region in regions {
            for nested in module.ops_in_region(region) {
                effects.extend(self.visit(module, nested));
            }
        }
        effects.sort_by_key(|e| format!("{e}"));
        effects.dedup();
        self.summaries.insert(op, effects.clone());
        effects
    }

    /// Every effect the operation performs, itself and its regions.
    pub fn of(&self, op: OperationId) -> &[Effect] {
        self.summaries.get(&op).map_or(&[], Vec::as_slice)
    }

    /// Whether the operation touches nothing outside the program.
    pub fn is_pure(&self, op: OperationId) -> bool {
        self.of(op).iter().all(Effect::is_pure)
    }

    /// Whether re-running the operation is observationally free.
    ///
    /// The precondition for *Dead Action Elimination*, *Result Reuse* and
    /// *Tool Call Deduplication* (§5).
    pub fn is_replayable(&self, op: OperationId) -> bool {
        self.of(op).iter().all(Effect::is_replayable)
    }

    /// Whether the operation can perform an effect that cannot be undone.
    ///
    /// *Speculative Execution* must never fire when this holds (§5, §7).
    pub fn is_irreversible(&self, op: OperationId) -> bool {
        self.of(op).iter().any(Effect::is_irreversible)
    }

    /// Whether the operation writes anything outside the program.
    pub fn writes(&self, op: OperationId) -> bool {
        self.of(op).iter().any(Effect::is_write)
    }

    /// Whether the operation's outcome is non-deterministic.
    pub fn is_stochastic(&self, op: OperationId) -> bool {
        self.of(op).iter().any(|e| matches!(e, Effect::Stochastic))
    }

    /// Whether two operations may run concurrently — invariant I3.
    ///
    /// True when some effect of one conflicts with some effect of the other,
    /// which is exactly "they write a scope the other also touches".
    pub fn conflict(&self, a: OperationId, b: OperationId) -> bool {
        self.of(a)
            .iter()
            .any(|left| self.of(b).iter().any(|right| left.conflicts_with(right)))
    }

    /// The effects of the operation that need a capability (§2.3, I2).
    pub fn capability_demanding(&self, op: OperationId) -> Vec<&Effect> {
        self.of(op)
            .iter()
            .filter(|e| e.requires_capability())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_parser::parse_module;

    const PROGRAM: &str = r#"
module @m version(0) {
  agent.func "f" {
  ^bb0(%input: !tool.ref<db>):
    %pure = agent.action "compute"(%input) {effect = #pure} : !core.int
    control.loop(%input) {
    ^bb0(%item: !tool.ref<db>):
      tool.call "delete_row"(%item) {effect = #irreversible<db>}
    } {effect = #pure, max_iterations = 4}
    %read = tool.call "fetch_row"(%input) {effect = #read_external<db>} : !tool.result<row>
    %other = tool.call "fetch_cache"(%input) {effect = #read_external<cache>} : !tool.result<row>
  } {effect = #pure}
}
"#;

    fn op_with_literal(module: &Module, literal: &str) -> OperationId {
        let mut found = None;
        module.walk(|op| {
            if op.literal.as_deref() == Some(literal) {
                found = Some(op.id);
            }
        });
        found.unwrap()
    }

    fn loop_op(module: &Module) -> OperationId {
        let mut found = None;
        module.walk(|op| {
            if op.name.is("control", "loop") {
                found = Some(op.id);
            }
        });
        found.unwrap()
    }

    #[test]
    fn a_pure_loop_over_an_irreversible_body_is_not_pure() {
        let module = parse_module(PROGRAM).unwrap();
        let summary = EffectSummary::build(&module);
        let looping = loop_op(&module);
        assert_eq!(module.op(looping).effect, agent_ir_core::Effect::Pure);
        assert!(!summary.is_pure(looping), "the declaration hid the body");
        assert!(summary.is_irreversible(looping));
        assert!(!summary.is_replayable(looping));
    }

    #[test]
    fn a_pure_action_stays_pure() {
        let module = parse_module(PROGRAM).unwrap();
        let summary = EffectSummary::build(&module);
        let compute = op_with_literal(&module, "compute");
        assert!(summary.is_pure(compute));
        assert!(summary.is_replayable(compute));
        assert!(summary.capability_demanding(compute).is_empty());
    }

    #[test]
    fn a_read_conflicts_with_an_irreversible_write_on_the_same_scope() {
        let module = parse_module(PROGRAM).unwrap();
        let summary = EffectSummary::build(&module);
        let looping = loop_op(&module);
        let read = op_with_literal(&module, "fetch_row");
        assert!(summary.conflict(looping, read));
    }

    #[test]
    fn two_reads_never_conflict() {
        let module = parse_module(PROGRAM).unwrap();
        let summary = EffectSummary::build(&module);
        let db = op_with_literal(&module, "fetch_row");
        let cache = op_with_literal(&module, "fetch_cache");
        assert!(!summary.conflict(db, cache));
    }

    #[test]
    fn the_function_summary_reaches_the_whole_body() {
        let module = parse_module(PROGRAM).unwrap();
        let summary = EffectSummary::build(&module);
        let func = module.function("f").unwrap();
        assert!(summary.is_irreversible(func));
        assert!(summary.writes(func));
        assert_eq!(summary.capability_demanding(func).len(), 3);
    }
}
