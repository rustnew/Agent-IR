//! The use map: for each value, every operand slot that reads it.
//!
//! *Dead Action Elimination* and *Dead Context Elimination* both reduce to a
//! question this structure answers, and answering it once beats each pass
//! walking the module again.

use agent_ir_core::{Module, OperationId, ValueId};

/// One read of a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Use {
    /// The operation doing the reading.
    pub op: OperationId,
    /// Which of its operands this is.
    pub operand: usize,
}

/// Every use of every value in a module.
#[derive(Clone, Debug)]
pub struct UseMap {
    uses: Vec<Vec<Use>>,
}

impl UseMap {
    /// Walks the live program and records every operand read.
    pub fn build(module: &Module) -> Self {
        let mut uses = vec![Vec::new(); module.value_capacity()];
        module.walk(|op| {
            for (index, &operand) in op.operands.iter().enumerate() {
                uses[operand.index()].push(Use { op: op.id, operand: index });
            }
        });
        UseMap { uses }
    }

    /// Every read of a value.
    pub fn uses_of(&self, value: ValueId) -> &[Use] {
        self.uses.get(value.index()).map_or(&[], Vec::as_slice)
    }

    /// Whether anything reads the value.
    pub fn is_used(&self, value: ValueId) -> bool {
        !self.uses_of(value).is_empty()
    }

    /// How many operand slots read the value.
    pub fn use_count(&self, value: ValueId) -> usize {
        self.uses_of(value).len()
    }

    /// The operations that read the value, without duplicates.
    pub fn readers(&self, value: ValueId) -> Vec<OperationId> {
        let mut readers: Vec<OperationId> = self.uses_of(value).iter().map(|u| u.op).collect();
        readers.dedup();
        readers
    }

    /// Whether none of an operation's results is read.
    ///
    /// This is the dataflow half of the *Dead Action Elimination* condition;
    /// the effect half lives in [`crate::EffectSummary`].
    pub fn results_are_unread(&self, module: &Module, op: OperationId) -> bool {
        module.op(op).results.iter().all(|&r| !self.is_used(r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_parser::parse_module;

    const PROGRAM: &str = r#"
module @m version(0) {
  agent.func "f" {
  ^bb0(%input: !tool.ref<x>):
    %used = agent.action "a"(%input) {effect = #pure} : !core.int
    %unused = agent.action "b"(%input) {effect = #pure} : !core.int
    %twice = agent.action "c"(%used, %used) {effect = #pure} : !core.int
    agent.return(%twice) {effect = #pure}
  } {effect = #pure}
}
"#;

    fn value(module: &Module, name: &str) -> ValueId {
        module
            .all_values()
            .find(|v| v.name.as_deref() == Some(name))
            .unwrap()
            .id
    }

    #[test]
    fn counts_every_operand_slot() {
        let module = parse_module(PROGRAM).unwrap();
        let map = UseMap::build(&module);
        assert_eq!(map.use_count(value(&module, "used")), 2);
        assert_eq!(map.readers(value(&module, "used")).len(), 1);
    }

    #[test]
    fn an_unread_result_is_unused() {
        let module = parse_module(PROGRAM).unwrap();
        let map = UseMap::build(&module);
        assert!(!map.is_used(value(&module, "unused")));
        assert!(map.is_used(value(&module, "twice")));
    }

    #[test]
    fn a_block_argument_read_three_times_is_tracked() {
        let module = parse_module(PROGRAM).unwrap();
        let map = UseMap::build(&module);
        assert_eq!(map.use_count(value(&module, "input")), 2);
    }
}
