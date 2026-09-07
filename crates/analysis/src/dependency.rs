//! The dependency graph over the operations of one block.
//!
//! Two operations must keep their relative order when either
//!
//! * the later one reads a value the earlier one produced (a data dependency),
//!   or
//! * their effects conflict — one writes a scope the other touches (I3).
//!
//! Everything else the LLM wrote down in sequence is sequential only by
//! accident of emission order, and is what *Parallelization* is allowed to
//! reorder. The graph is the evidence for that claim; the pass never guesses.

use crate::effects::EffectSummary;
use agent_ir_core::{BlockId, Module, OperationId};
use std::collections::{BTreeSet, HashMap};

/// Which operations of a block depend on which.
#[derive(Clone, Debug)]
pub struct DependencyGraph {
    block: BlockId,
    order: Vec<OperationId>,
    /// For each operation, the earlier operations it directly depends on.
    predecessors: HashMap<OperationId, BTreeSet<OperationId>>,
    /// Why each edge exists, for reporting and for tests.
    reasons: HashMap<(OperationId, OperationId), Reason>,
}

/// Why one operation must follow another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The later operation reads a value the earlier one produced.
    Data,
    /// Their effects conflict on an overlapping scope (I3).
    Effect,
    /// Both: a data dependency that is also an effect conflict.
    Both,
}

impl DependencyGraph {
    /// Builds the graph for the operations directly inside a block.
    ///
    /// Operations nested deeper are represented by the operation that owns
    /// their region, using its effect summary, so a `control.loop` that deletes
    /// rows is correctly ordered against a read of the same table.
    pub fn of_block(module: &Module, block: BlockId, effects: &EffectSummary) -> Self {
        let order: Vec<OperationId> = module
            .block(block)
            .ops
            .iter()
            .copied()
            .filter(|&op| !module.op(op).erased)
            .collect();

        let mut predecessors: HashMap<OperationId, BTreeSet<OperationId>> =
            order.iter().map(|&op| (op, BTreeSet::new())).collect();
        let mut reasons = HashMap::new();

        // A value produced anywhere inside an operation's subtree is attributed
        // to that operation, so a use of a nested result still creates an edge.
        let mut producer = HashMap::new();
        for &op in &order {
            collect_produced(module, op, op, &mut producer);
        }

        for (position, &later) in order.iter().enumerate() {
            let mut reads = Vec::new();
            collect_reads(module, later, &mut reads);

            for &earlier in &order[..position] {
                let data = reads.iter().any(|value| producer.get(value) == Some(&earlier));
                let effect = effects.conflict(earlier, later);
                let reason = match (data, effect) {
                    (true, true) => Some(Reason::Both),
                    (true, false) => Some(Reason::Data),
                    (false, true) => Some(Reason::Effect),
                    (false, false) => None,
                };
                if let Some(reason) = reason {
                    predecessors.get_mut(&later).unwrap().insert(earlier);
                    reasons.insert((earlier, later), reason);
                }
            }
        }

        DependencyGraph { block, order, predecessors, reasons }
    }

    /// The block this graph describes.
    pub fn block(&self) -> BlockId {
        self.block
    }

    /// The operations, in program order.
    pub fn operations(&self) -> &[OperationId] {
        &self.order
    }

    /// The operations `op` directly depends on.
    pub fn predecessors(&self, op: OperationId) -> impl Iterator<Item = OperationId> + '_ {
        self.predecessors.get(&op).into_iter().flatten().copied()
    }

    /// Why `later` must follow `earlier`, if it must.
    pub fn reason(&self, earlier: OperationId, later: OperationId) -> Option<Reason> {
        self.reasons.get(&(earlier, later)).copied()
    }

    /// Whether `later` must follow `earlier`, directly or transitively.
    pub fn depends_on(&self, later: OperationId, earlier: OperationId) -> bool {
        let mut stack = vec![later];
        let mut seen = BTreeSet::new();
        while let Some(current) = stack.pop() {
            for predecessor in self.predecessors(current) {
                if predecessor == earlier {
                    return true;
                }
                if seen.insert(predecessor) {
                    stack.push(predecessor);
                }
            }
        }
        false
    }

    /// Groups the operations into the batches the scheduler of §15 executes.
    ///
    /// Level *n* holds every operation whose longest dependency chain is *n*
    /// long, so everything in one level may run concurrently and the levels run
    /// in order. Program order is preserved inside a level, which keeps the
    /// output stable and the printed IR readable.
    pub fn levels(&self) -> Vec<Vec<OperationId>> {
        let mut depth: HashMap<OperationId, usize> = HashMap::new();
        for &op in &self.order {
            let level = self
                .predecessors(op)
                .map(|p| depth.get(&p).copied().unwrap_or(0) + 1)
                .max()
                .unwrap_or(0);
            depth.insert(op, level);
        }
        let height = depth.values().copied().max().map_or(0, |m| m + 1);
        let mut levels = vec![Vec::new(); height];
        for &op in &self.order {
            levels[depth[&op]].push(op);
        }
        levels
    }

    /// Whether the whole block is one sequential chain, with nothing to gain
    /// from parallel scheduling.
    pub fn is_fully_sequential(&self) -> bool {
        self.levels().iter().all(|level| level.len() <= 1)
    }
}

/// Records every value produced inside `op`'s subtree as produced by `root`.
fn collect_produced(
    module: &Module,
    root: OperationId,
    op: OperationId,
    out: &mut HashMap<agent_ir_core::ValueId, OperationId>,
) {
    for &result in &module.op(op).results {
        out.insert(result, root);
    }
    for &region in &module.op(op).regions {
        for &block in &module.region(region).blocks {
            for &arg in &module.block(block).args {
                out.insert(arg, root);
            }
            for &nested in &module.block(block).ops {
                if !module.op(nested).erased {
                    collect_produced(module, root, nested, out);
                }
            }
        }
    }
}

/// Every value read inside `op`'s subtree.
fn collect_reads(module: &Module, op: OperationId, out: &mut Vec<agent_ir_core::ValueId>) {
    out.extend(module.op(op).operands.iter().copied());
    for &region in &module.op(op).regions {
        for &block in &module.region(region).blocks {
            for &nested in &module.block(block).ops {
                if !module.op(nested).erased {
                    collect_reads(module, nested, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_parser::parse_module;

    fn graph_of(source: &str) -> (Module, DependencyGraph) {
        let module = parse_module(source).unwrap();
        let func = module.function("f").unwrap();
        let region = module.op(func).regions[0];
        let block = module.region(region).blocks[0];
        let effects = EffectSummary::build(&module);
        let graph = DependencyGraph::of_block(&module, block, &effects);
        (module, graph)
    }

    fn op_with_literal(module: &Module, literal: &str) -> OperationId {
        let mut found = None;
        module.walk(|op| {
            if op.literal.as_deref() == Some(literal) {
                found = Some(op.id);
            }
        });
        found.unwrap_or_else(|| panic!("no operation `{literal}`"))
    }

    const INDEPENDENT: &str = r#"
module @m version(0) {
  agent.func "f" {
  ^bb0(%a: !tool.ref<x>, %b: !tool.ref<y>, %c: !tool.ref<z>):
    %ia = agent.action "inspect_model"(%a) {effect = #pure} : !core.int
    %ib = agent.action "inspect_hardware"(%b) {effect = #pure} : !core.int
    %ic = agent.action "inspect_dataset"(%c) {effect = #pure} : !core.int
    %p = agent.action "profile"(%ia, %ib, %ic) {effect = #read_external<host>} : !core.int
    agent.return(%p) {effect = #pure}
  } {effect = #pure}
}
"#;

    #[test]
    fn independent_pure_actions_have_no_edges_between_them() {
        let (module, graph) = graph_of(INDEPENDENT);
        let a = op_with_literal(&module, "inspect_model");
        let b = op_with_literal(&module, "inspect_hardware");
        assert!(graph.reason(a, b).is_none());
        assert!(!graph.depends_on(b, a));
    }

    #[test]
    fn a_consumer_depends_on_all_three_producers() {
        let (module, graph) = graph_of(INDEPENDENT);
        let profile = op_with_literal(&module, "profile");
        for literal in ["inspect_model", "inspect_hardware", "inspect_dataset"] {
            let producer = op_with_literal(&module, literal);
            assert_eq!(graph.reason(producer, profile), Some(Reason::Data), "{literal}");
        }
    }

    #[test]
    fn the_three_inspections_land_in_one_level() {
        let (_, graph) = graph_of(INDEPENDENT);
        let levels = graph.levels();
        assert_eq!(levels[0].len(), 3, "the inspections should batch together");
        assert_eq!(levels[1].len(), 1, "profile depends on all three");
        assert_eq!(levels[2].len(), 1, "return depends on profile");
        assert!(!graph.is_fully_sequential());
    }

    const CONFLICTING: &str = r#"
module @m version(0) {
  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    tool.call "write_row"(%row) {effect = #write_external<db>}
    %r = tool.call "read_row"(%row) {effect = #read_external<db>} : !tool.result<row>
    %elsewhere = tool.call "read_cache"(%row) {effect = #read_external<cache>} : !tool.result<row>
    agent.return(%r) {effect = #pure}
  } {effect = #pure}
}
"#;

    #[test]
    fn a_write_orders_a_later_read_of_the_same_scope() {
        let (module, graph) = graph_of(CONFLICTING);
        let write = op_with_literal(&module, "write_row");
        let read = op_with_literal(&module, "read_row");
        assert_eq!(graph.reason(write, read), Some(Reason::Effect));
    }

    #[test]
    fn a_write_leaves_another_scope_free_to_move() {
        let (module, graph) = graph_of(CONFLICTING);
        let write = op_with_literal(&module, "write_row");
        let cache = op_with_literal(&module, "read_cache");
        assert!(graph.reason(write, cache).is_none());
    }

    const NESTED_EFFECT: &str = r#"
module @m version(0) {
  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    control.loop(%row) {
    ^bb0(%item: !tool.ref<db>):
      tool.call "delete_row"(%item) {effect = #irreversible<db>}
    } {effect = #pure, max_iterations = 4}
    %r = tool.call "read_row"(%row) {effect = #read_external<db>} : !tool.result<row>
    agent.return(%r) {effect = #pure}
  } {effect = #pure}
}
"#;

    #[test]
    fn an_effect_hidden_in_a_region_still_orders_the_block() {
        let (module, graph) = graph_of(NESTED_EFFECT);
        let looping = graph.operations()[0];
        assert!(module.op(looping).name.is("control", "loop"));
        let read = op_with_literal(&module, "read_row");
        assert_eq!(graph.reason(looping, read), Some(Reason::Effect));
        assert!(graph.is_fully_sequential());
    }
}
