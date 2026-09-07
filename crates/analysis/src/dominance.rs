//! Dominance over structured control flow (invariant I1).
//!
//! Agent IR v0.1 has no arbitrary CFG: control flow is `control.if`,
//! `control.while`, `control.loop` and `control.parallel`, each owning regions.
//! Dominance therefore collapses to a lexical question — is the definition in
//! an enclosing block, at an earlier position? — which is both cheaper and
//! easier to get right than a dominator tree.

use agent_ir_core::{BlockId, Module, OperationId, ValueDef, ValueId};

/// One step of the chain from an operation out to the module body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    /// A block enclosing the operation.
    pub block: BlockId,
    /// The position, inside that block, of the operation or of the enclosing
    /// operation that contains it.
    pub position: usize,
}

/// The blocks enclosing an operation, innermost first.
///
/// Each frame records where the chain passes through that block, which is what
/// makes "defined earlier" answerable at every level at once.
pub fn enclosing_chain(module: &Module, op: OperationId) -> Vec<Frame> {
    let mut chain = Vec::new();
    let mut current = op;
    while let Some(block) = module.op(current).parent {
        let Some(position) = module.position_in_block(current) else {
            break;
        };
        chain.push(Frame { block, position });
        match module.region(module.block(block).parent).parent {
            Some(parent) => current = parent,
            None => break,
        }
    }
    chain
}

/// Whether `value` is defined before `op` on every path that reaches it.
///
/// A block argument dominates everything in its block and in the regions
/// nested inside it. An operation result dominates an operation that appears
/// later in the same block, or anywhere inside a region that operation owns.
pub fn dominates(module: &Module, value: ValueId, op: OperationId) -> bool {
    let chain = enclosing_chain(module, op);
    match module.value(value).def {
        ValueDef::BlockArg { block, .. } => chain.iter().any(|frame| frame.block == block),
        ValueDef::OpResult { op: producer, .. } => {
            if module.op(producer).erased {
                return false;
            }
            let Some(producer_block) = module.op(producer).parent else {
                return false;
            };
            let Some(producer_position) = module.position_in_block(producer) else {
                return false;
            };
            chain
                .iter()
                .find(|frame| frame.block == producer_block)
                .is_some_and(|frame| producer_position < frame.position)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_parser::parse_module;

    fn module_of(source: &str) -> Module {
        parse_module(source).expect("test program should parse")
    }

    fn value(module: &Module, name: &str) -> ValueId {
        module
            .all_values()
            .find(|v| v.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no value named `{name}`"))
            .id
    }

    fn op_with_literal(module: &Module, literal: &str) -> OperationId {
        let mut found = None;
        module.walk(|op| {
            if op.literal.as_deref() == Some(literal) {
                found = Some(op.id);
            }
        });
        found.unwrap_or_else(|| panic!("no operation with literal `{literal}`"))
    }

    const NESTED: &str = r#"
module @m version(0) {
  agent.func "f" {
  ^bb0(%input: !tool.ref<x>):
    %outer = agent.action "make"(%input) {effect = #pure} : !core.int
    control.loop(%input) {
    ^bb0(%item: !tool.ref<x>):
      %inner = agent.action "use"(%outer, %item) {effect = #pure} : !core.int
    } {effect = #pure, max_iterations = 4}
    %after = agent.action "later"(%outer) {effect = #pure} : !core.int
  } {effect = #pure}
}
"#;

    #[test]
    fn an_outer_definition_dominates_a_nested_use() {
        let module = module_of(NESTED);
        let outer = value(&module, "outer");
        let inner_use = op_with_literal(&module, "use");
        assert!(dominates(&module, outer, inner_use));
    }

    #[test]
    fn a_block_argument_dominates_its_own_region() {
        let module = module_of(NESTED);
        let item = value(&module, "item");
        let inner_use = op_with_literal(&module, "use");
        assert!(dominates(&module, item, inner_use));
    }

    #[test]
    fn a_nested_definition_does_not_dominate_an_outer_use() {
        let module = module_of(NESTED);
        let inner = value(&module, "inner");
        let later = op_with_literal(&module, "later");
        assert!(!dominates(&module, inner, later));
    }

    #[test]
    fn a_later_definition_does_not_dominate_an_earlier_use() {
        let module = module_of(NESTED);
        let after = value(&module, "after");
        let make = op_with_literal(&module, "make");
        assert!(!dominates(&module, after, make));
    }

    #[test]
    fn the_chain_reaches_the_module_body() {
        let module = module_of(NESTED);
        let inner_use = op_with_literal(&module, "use");
        let chain = enclosing_chain(&module, inner_use);
        // loop body block, function entry block, module body block.
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.last().unwrap().block, module.body_block());
    }
}
