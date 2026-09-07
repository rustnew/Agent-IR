//! Dead Action Elimination (§5).
//!
//! **Validity condition.** An operation may be removed when
//!
//! 1. it produces at least one result and none of them is read, and
//! 2. its *effect summary* — itself plus everything in its regions — is
//!    replayable, that is every effect is `Pure` or `ReadExternal`.
//!
//! **Risk if violated.** Removing a silently useful side effect: a log line, a
//! cache warm-up, a write whose value nobody reads but whose happening matters.
//! Condition 2 is what keeps that from happening, and condition 1 is what keeps
//! the pass from deleting `agent.return`, `agent.verify`, `memory.write` and
//! every other operation that exists purely for its effect.
//!
//! §5 lists *Dead Context Elimination* as a separate entry. It is the same
//! transformation restricted to context-carrying values, so it is implemented
//! here; the report distinguishes the two so the two rows of the table stay
//! measurable apart.

use crate::manager::{Pass, PassReport};
use agent_ir_analysis::{EffectSummary, UseMap};
use agent_ir_core::{Diagnostic, Module, OperationId};

/// Removes operations whose results nobody reads and whose effects are
/// replayable.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeadActionElimination;

impl Pass for DeadActionElimination {
    fn name(&self) -> &'static str {
        "dead-action-elimination"
    }

    fn description(&self) -> &'static str {
        "Removes operations whose results are unread, when their effect summary is replayable."
    }

    fn run(&self, module: &mut Module) -> PassReport {
        let mut report = PassReport::new(self.name());

        // Removing one operation can make its operands' producers dead in turn,
        // so the pass iterates until nothing more is removable.
        loop {
            let uses = UseMap::build(module);
            let effects = EffectSummary::build(module);

            let dead: Vec<OperationId> = module
                .op_ids()
                .into_iter()
                .filter(|&op| is_dead(module, op, &uses, &effects))
                .collect();

            if dead.is_empty() {
                break;
            }

            for op in dead {
                note_removal(module, op, &effects, &mut report);
                module.erase(op);
                report.removed(op);
            }
        }

        report
    }
}

fn is_dead(
    module: &Module,
    op: OperationId,
    uses: &UseMap,
    effects: &EffectSummary,
) -> bool {
    let operation = module.op(op);

    // An operation with no results exists for its effect. Even a `#pure` one —
    // `agent.verify` and `agent.reject` produce nothing and must survive.
    if operation.results.is_empty() {
        return false;
    }

    // A function is reachable from outside the module.
    if operation.name.is("agent", "func") {
        return false;
    }

    // Condition 1: nothing reads what it produces.
    if !uses.results_are_unread(module, op) {
        return false;
    }

    // Condition 2: re-running it, or never running it, is observationally free.
    effects.is_replayable(op)
}

/// Records which row of the §5 table this removal belongs to.
fn note_removal(
    module: &Module,
    op: OperationId,
    effects: &EffectSummary,
    report: &mut PassReport,
) {
    let operation = module.op(op);
    let is_context = operation.name.is("agent", "context")
        || operation.name.is("core", "constant")
        || operation.name.is("memory", "read");
    let rule = if is_context { "dead-context" } else { "dead-action" };
    let kind = if effects.is_pure(op) { "pure" } else { "read-only" };
    report.notes.push(
        Diagnostic::note(
            rule,
            format!(
                "removed {kind} `{}`{}: its result is never read",
                operation.name,
                operation
                    .literal
                    .as_deref()
                    .map(|l| format!(" \"{l}\""))
                    .unwrap_or_default()
            ),
        )
        .at(op),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::Pass;
    use agent_ir_parser::parse_module;

    fn run(source: &str) -> (Module, PassReport) {
        let mut module = parse_module(source).expect("fixture should parse");
        let report = DeadActionElimination.run(&mut module);
        (module, report)
    }

    #[test]
    fn removes_an_unread_pure_action() {
        let (module, report) = run(
            r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%x: !core.int):
    %unused = agent.action "compute"(%x) {effect = #pure} : !core.int
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert_eq!(report.removed.len(), 1);
        assert!(!module.to_string().contains("compute"));
    }

    #[test]
    fn removes_an_unread_external_read() {
        // §5 allows this: `ReadExternal` with a result proven unused.
        let (module, report) = run(
            r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert_eq!(report.removed.len(), 1);
        assert!(!module.to_string().contains("fetch"));
    }

    #[test]
    fn refuses_to_remove_an_unread_write() {
        let (module, report) = run(
            r#"module @m version(0) {
  capability @db scope("db") grants(write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    %receipt = tool.call "insert"(%row) {effect = #write_external<db>} : !tool.result<receipt>
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert!(!report.changed, "a write is not dead just because nobody reads its receipt");
        assert!(module.to_string().contains("insert"));
    }

    #[test]
    fn refuses_to_remove_an_unread_irreversible_action() {
        let (module, report) = run(
            r#"module @m version(0) {
  capability @pay scope("ledger") grants(irreversible)

  agent.func "f" {
  ^bb0(%invoice: !tool.ref<invoice>):
    agent.verify(%invoice) {effect = #pure, capability = "pay"}
    %receipt = tool.call "pay_invoice"(%invoice) {effect = #irreversible<ledger>} : !tool.result<receipt>
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert!(!report.changed);
        assert!(module.to_string().contains("pay_invoice"));
    }

    #[test]
    fn refuses_to_remove_a_pure_looking_operation_that_hides_an_effect() {
        // The `control.parallel` line says `#pure`. Its region deletes rows.
        let (module, report) = run(
            r#"module @m version(0) {
  capability @db scope("db") grants(irreversible)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    agent.verify(%row) {effect = #pure, capability = "db"}
    %ignored = control.parallel {
      tool.call "delete_row"(%row) {effect = #irreversible<db>}
      %n = core.constant {effect = #pure, value = 1} : !core.int
      control.yield(%n) {effect = #pure}
    } {effect = #pure} : !core.int
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert!(!report.changed, "the declaration hid the body:\n{module}");
        assert!(module.to_string().contains("delete_row"));
    }

    #[test]
    fn cascades_through_a_chain_of_dead_operations() {
        let (module, report) = run(
            r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%x: !core.int):
    %a = agent.action "one"(%x) {effect = #pure} : !core.int
    %b = agent.action "two"(%a) {effect = #pure} : !core.int
    %c = agent.action "three"(%b) {effect = #pure} : !core.int
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert_eq!(report.removed.len(), 3, "the whole chain should go");
        let printed = module.to_string();
        for literal in ["one", "two", "three"] {
            assert!(!printed.contains(literal), "{literal} survived:\n{printed}");
        }
    }

    #[test]
    fn keeps_everything_that_feeds_the_return() {
        let (_, report) = run(
            r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%x: !core.int):
    %a = agent.action "one"(%x) {effect = #pure} : !core.int
    %b = agent.action "two"(%a) {effect = #pure} : !core.int
    agent.return(%b) {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert!(!report.changed);
    }

    #[test]
    fn never_removes_an_operation_that_produces_nothing() {
        let (module, report) = run(
            r#"module @m version(0) {
  capability @db scope("db") grants(write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    memory.write(%row) {effect = #write_external<db>, key = "last_row"}
    agent.reject(%row) {effect = #pure}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        assert!(!report.changed);
        assert!(module.to_string().contains("memory.write"));
        assert!(module.to_string().contains("agent.reject"));
    }

    #[test]
    fn distinguishes_dead_context_from_dead_action_in_the_report() {
        let (_, report) = run(
            r#"module @m version(0) {
  agent.func "f" {
    %k = core.constant {effect = #pure, value = 1} : !core.int
    %a = agent.action "compute"() {effect = #pure} : !core.int
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        );
        let codes: Vec<&str> = report.notes.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"dead-context"), "{codes:?}");
        assert!(codes.contains(&"dead-action"), "{codes:?}");
    }
}
