//! Parallelization (§5, §13.1).
//!
//! **Validity condition.** A run of operations may share a `control.parallel`
//! region when no two of them are connected in the dependency graph — neither
//! by data, nor by an effect conflict on an overlapping scope. That second half
//! is invariant I3, and the effect summary supplies it, so an operation whose
//! own line says `#pure` while its region deletes rows is still ordered
//! correctly.
//!
//! **Risk if violated.** A race on a shared external resource.
//!
//! ## Why only adjacent operations
//!
//! The pass fuses a run of operations that are *already adjacent* in the block.
//! It never reorders. That is a real limitation — a program written
//! `A, B(dep A), C, D(dep C)` has `A` and `C` independent, and a reordering
//! scheduler would batch them — and it is deliberate.
//!
//! Moving an operation across one it does not depend on is not safe in general
//! from the dependency graph alone: shifting an operation earlier can jump it
//! over one of its own transitive predecessors, and shifting it later can jump
//! it over one of its successors. Getting that right needs an interval analysis
//! this pass does not have, and §5 is explicit that a pass fires only when its
//! condition is *satisfied*, not when it is plausible. Fusing adjacent runs
//! needs no such proof: the relative order of everything else is untouched.
//!
//! The scheduler of §15 recovers the rest of the parallelism at execution time,
//! where it can see the real dependency levels without rewriting the program.

use crate::manager::{Pass, PassReport};
use agent_ir_analysis::{DependencyGraph, EffectSummary, UseMap};
use agent_ir_core::{
    Attributes, BlockId, Diagnostic, Effect, Module, OpName, OperationId, Provenance, ValueId,
};

/// The shortest run worth wrapping. A "parallel" region around one operation
/// only adds noise.
const MIN_GROUP: usize = 2;

/// Wraps adjacent independent operations in a `control.parallel` region.
#[derive(Clone, Copy, Debug, Default)]
pub struct Parallelization;

impl Pass for Parallelization {
    fn name(&self) -> &'static str {
        "parallelization"
    }

    fn description(&self) -> &'static str {
        "Groups adjacent operations with no dependency and no effect conflict into control.parallel."
    }

    fn run(&self, module: &mut Module) -> PassReport {
        let mut report = PassReport::new(self.name());

        for block in collect_blocks(module) {
            // One group per visit: wrapping changes the block, so the graph is
            // rebuilt and the block re-examined until nothing more groups.
            while let Some(group) = next_group(module, block) {
                wrap(module, block, &group, &mut report);
            }
        }

        report
    }
}

/// Every block worth scheduling.
///
/// A block that *is* the body of a `control.parallel` is skipped: its
/// operations already run concurrently, so nesting another region inside would
/// only deepen the IR without changing what executes when.
fn collect_blocks(module: &Module) -> Vec<BlockId> {
    let mut blocks = Vec::new();
    for index in 0..module.region_count() {
        let region = module.region_by_index(index);
        let already_parallel = region
            .parent
            .is_some_and(|owner| module.op(owner).name.is("control", "parallel"));
        if already_parallel {
            continue;
        }
        blocks.extend(region.blocks.iter().copied());
    }
    blocks
}

/// The next run of adjacent, mutually independent operations in the block.
fn next_group(module: &Module, block: BlockId) -> Option<Vec<OperationId>> {
    let effects = EffectSummary::build(module);
    let graph = DependencyGraph::of_block(module, block, &effects);
    let ops = graph.operations();

    let mut start = 0;
    while start < ops.len() {
        if !is_groupable(module, ops[start]) {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < ops.len()
            && is_groupable(module, ops[end])
            && ops[start..end]
                .iter()
                .all(|&earlier| graph.reason(earlier, ops[end]).is_none())
        {
            end += 1;
        }
        if end - start >= MIN_GROUP {
            return Some(ops[start..end].to_vec());
        }
        start = if end > start { end } else { start + 1 };
    }
    None
}

/// Whether an operation may be moved inside a `control.parallel` region.
fn is_groupable(module: &Module, op: OperationId) -> bool {
    let operation = module.op(op);
    // A terminator ends its block and cannot be nested.
    if operation.name.is("agent", "return") || operation.name.is("control", "yield") {
        return false;
    }
    // Wrapping a `control.parallel` in another one gains nothing.
    if operation.name.is("control", "parallel") {
        return false;
    }
    // A function body is not a schedulable action.
    if operation.name.is("agent", "func") {
        return false;
    }
    // `agent.verify` guards what follows it (I2, I4). Moving it into a region
    // that runs concurrently with the action it guards would defeat the guard.
    if operation.name.is("agent", "verify") {
        return false;
    }
    true
}

/// Moves `group` into a fresh `control.parallel` at the position of its first
/// member, routing every value that escapes through a `control.yield`.
fn wrap(module: &mut Module, block: BlockId, group: &[OperationId], report: &mut PassReport) {
    let position = module
        .position_in_block(group[0])
        .expect("a grouped operation is in its block");

    // Which results are read from outside the group? Those, and only those,
    // have to leave the region.
    let uses = UseMap::build(module);
    let escaping: Vec<ValueId> = group
        .iter()
        .flat_map(|&op| module.op(op).results.clone())
        .filter(|&value| {
            uses.uses_of(value)
                .iter()
                .any(|use_site| !is_inside_group(module, use_site.op, group))
        })
        .collect();

    let result_types: Vec<(Option<String>, agent_ir_core::Type, Provenance)> = escaping
        .iter()
        .map(|&value| {
            let inner = module.value(value);
            (
                inner.name.clone(),
                inner.ty.clone(),
                inner.provenance.clone(),
            )
        })
        .collect();

    let parallel = module.create_op(
        OpName::new("control", "parallel"),
        None,
        Vec::new(),
        result_types,
        Attributes::new(),
        Effect::Pure,
    );
    module.insert_in_block(block, position, parallel);
    report.created(parallel);

    let region = module.create_op_region(parallel);
    let inner_block = module.create_block(region);
    for &op in group {
        module.move_to_end(op, inner_block);
    }

    // Rewrite the outside world to read the region's results. This must happen
    // before the yield exists, or the yield would be rewritten to read the very
    // values it is supposed to produce.
    let outputs = module.op(parallel).results.clone();
    for (&from, &to) in escaping.iter().zip(outputs.iter()) {
        module.value_mut(from).name = module
            .value(from)
            .name
            .clone()
            .map(|name| format!("{name}_local"));
        module.replace_all_uses(from, to);
        report.rewrote(from, to);
    }

    if !escaping.is_empty() {
        let yielded = module.create_op(
            OpName::new("control", "yield"),
            None,
            escaping.clone(),
            Vec::new(),
            Attributes::new(),
            Effect::Pure,
        );
        module.append_to_block(inner_block, yielded);
        report.created(yielded);
    }

    report.notes.push(
        Diagnostic::note(
            "parallelization",
            format!(
                "grouped {} adjacent operations with no dependency and no effect conflict",
                group.len()
            ),
        )
        .at(parallel),
    );
}

/// Whether an operation is one of the group, or nested inside one.
fn is_inside_group(module: &Module, op: OperationId, group: &[OperationId]) -> bool {
    if group.contains(&op) {
        return true;
    }
    module
        .ancestors(op)
        .iter()
        .any(|ancestor| group.contains(ancestor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_parser::parse_module;
    use agent_ir_verifier::Verifier;

    fn run(source: &str) -> (Module, PassReport) {
        let mut module = parse_module(source).expect("fixture should parse");
        let report = Parallelization.run(&mut module);
        let check = Verifier::new().verify(&module);
        assert!(
            !check.has_errors(),
            "the pass produced an invalid module:\n{module}\n{check}"
        );
        (module, report)
    }

    const SEQUENTIAL: &str = r#"module @m version(0) {
  capability @host scope("host") grants(read_external)

  agent.func "f" {
  ^bb0(%model: !tool.ref<model>, %hardware: !tool.ref<hw>, %dataset: !tool.ref<dataset>):
    %a = agent.action "inspect_model"(%model) {effect = #pure} : !core.int
    %b = agent.action "inspect_hardware"(%hardware) {effect = #pure} : !core.int
    %c = agent.action "inspect_dataset"(%dataset) {effect = #pure} : !core.int
    %p = agent.action "profile"(%a, %b, %c) {effect = #read_external<host>} : !core.int
    agent.return(%p) {effect = #pure}
  } {effect = #pure}
}
"#;

    #[test]
    fn groups_the_three_independent_inspections_of_section_13_1() {
        let (module, report) = run(SEQUENTIAL);
        let printed = module.to_string();
        assert!(report.changed);
        assert!(printed.contains("control.parallel"), "{printed}");
        assert_eq!(printed.matches("control.parallel").count(), 1);
        // All three inspections moved inside, and the profile still reads them.
        let region_start = printed.find("control.parallel").unwrap();
        let region_end = printed.find("control.yield").unwrap();
        let region = &printed[region_start..region_end];
        for literal in ["inspect_model", "inspect_hardware", "inspect_dataset"] {
            assert!(
                region.contains(literal),
                "{literal} not in region:\n{printed}"
            );
        }
        assert!(!region.contains("profile"));
    }

    #[test]
    fn the_three_results_leave_through_a_yield() {
        let (module, _) = run(SEQUENTIAL);
        let printed = module.to_string();
        let yield_line = printed
            .lines()
            .find(|l| l.contains("control.yield"))
            .unwrap();
        assert_eq!(yield_line.matches('%').count(), 3, "{yield_line}");
    }

    #[test]
    fn refuses_to_group_a_producer_with_its_consumer() {
        let (_, report) = run(r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%x: !core.int):
    %a = agent.action "one"(%x) {effect = #pure} : !core.int
    %b = agent.action "two"(%a) {effect = #pure} : !core.int
    agent.return(%b) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(!report.changed);
    }

    #[test]
    fn refuses_to_group_operations_that_conflict_on_a_scope() {
        let (_, report) = run(r#"module @m version(0) {
  capability @db scope("db") grants(read_external, write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    tool.call "update"(%row) {effect = #write_external<db>}
    %seen = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    agent.return(%seen) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(!report.changed, "I3 forbids this grouping");
    }

    #[test]
    fn groups_reads_of_different_scopes() {
        let (module, report) = run(r#"module @m version(0) {
  capability @db scope("db") grants(read_external)
  capability @cache scope("cache") grants(read_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    %a = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    %b = tool.call "peek"(%row) {effect = #read_external<cache>} : !tool.result<row>
    agent.return(%a) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(report.changed);
        assert!(module.to_string().contains("control.parallel"));
    }

    #[test]
    fn never_moves_a_verify_away_from_what_it_guards() {
        let (module, _) = run(r#"module @m version(0) {
  capability @pay scope("ledger") grants(irreversible)

  agent.func "f" {
  ^bb0(%invoice: !tool.ref<invoice>):
    agent.verify(%invoice) {effect = #pure, capability = "pay"}
    tool.call "pay"(%invoice) {effect = #irreversible<ledger>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#);
        let printed = module.to_string();
        assert!(!printed.contains("control.parallel"), "{printed}");
    }

    #[test]
    fn an_operation_with_no_escaping_result_needs_no_yield() {
        let (module, report) = run(r#"module @m version(0) {
  capability @log scope("log") grants(write_external)
  capability @audit scope("audit") grants(write_external)

  agent.func "f" {
  ^bb0(%event: !core.string):
    memory.write(%event) {effect = #write_external<log>, key = "log"}
    memory.write(%event) {effect = #write_external<audit>, key = "audit"}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(report.changed);
        let printed = module.to_string();
        assert!(printed.contains("control.parallel"));
        assert!(!printed.contains("control.yield"), "{printed}");
    }

    #[test]
    fn the_result_is_stable_under_a_second_run() {
        let mut module = parse_module(SEQUENTIAL).unwrap();
        Parallelization.run(&mut module);
        let once = module.to_string();
        let second = Parallelization.run(&mut module);
        assert!(!second.changed, "the pass is not idempotent:\n{module}");
        assert_eq!(once, module.to_string());
    }

    #[test]
    fn nested_blocks_are_scheduled_too() {
        let (module, report) = run(r#"module @m version(0) {
  capability @host scope("host") grants(read_external)

  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %a = agent.action "left"(%item) {effect = #pure} : !core.int
      %b = agent.action "right"(%item) {effect = #pure} : !core.int
      %c = agent.action "join"(%a, %b) {effect = #read_external<host>} : !core.int
    } {effect = #pure, max_iterations = 8}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(report.changed);
        assert!(module.to_string().contains("control.parallel"));
    }
}
