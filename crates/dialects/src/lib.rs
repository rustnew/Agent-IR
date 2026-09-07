//! # Agent IR dialects
//!
//! The six v0.1 dialects of §3.2, and the registry the verifier consults to
//! decide whether an operation is well formed.
//!
//! A dialect here is data, not code: each operation declares how many operands,
//! results and regions it takes, which attributes it needs, and which effect
//! classes it is allowed to declare. That last column is what stops a program
//! from labelling a payment `#pure` and slipping past every §5 pass condition.
//!
//! ## Two documented departures from §3.2
//!
//! * `core.cmp` is added. §3.3 writes `control.if (%accuracy < %threshold)`,
//!   and without a comparison operation there is no way to produce the boolean
//!   that `control.if` consumes.
//! * `control.branch` is not implemented in v0.1. Everything else in the
//!   dialect is structured control flow, and keeping it that way is what lets
//!   the effect analysis of §5 reason about a region as a unit instead of
//!   solving dataflow over an arbitrary CFG.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use agent_ir_core::{EffectClass, OpName};
use std::collections::BTreeMap;

/// The six dialect names of §3.2.
pub mod names {
    /// Module-level plumbing: constants, casts, comparisons.
    pub const CORE: &str = "core";
    /// The agent program itself: functions, context, actions, plans.
    pub const AGENT: &str = "agent";
    /// Structured control flow.
    pub const CONTROL: &str = "control";
    /// Tool invocation and capability checks.
    pub const TOOL: &str = "tool";
    /// The memory store.
    pub const MEMORY: &str = "memory";
    /// Observations and metrics fed back into the IR.
    pub const OBSERVATION: &str = "observation";

    /// Every dialect name, in specification order.
    pub const ALL: &[&str] = &[CORE, AGENT, CONTROL, TOOL, MEMORY, OBSERVATION];
}

/// How many of something an operation accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    /// Exactly this many.
    Exact(usize),
    /// This many or more.
    AtLeast(usize),
    /// Between the two bounds, inclusive.
    Between(usize, usize),
    /// Any number, including none.
    Any,
}

impl Arity {
    /// Whether `count` satisfies this arity.
    pub fn accepts(self, count: usize) -> bool {
        match self {
            Arity::Exact(n) => count == n,
            Arity::AtLeast(n) => count >= n,
            Arity::Between(lo, hi) => (lo..=hi).contains(&count),
            Arity::Any => true,
        }
    }

    /// A human description, for diagnostics.
    pub fn describe(self) -> String {
        match self {
            Arity::Exact(n) => format!("exactly {n}"),
            Arity::AtLeast(n) => format!("at least {n}"),
            Arity::Between(lo, hi) => format!("between {lo} and {hi}"),
            Arity::Any => "any number of".to_string(),
        }
    }
}

/// The kind an attribute must have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrKind {
    /// An integer.
    Int,
    /// A float, or an integer widened to one.
    Float,
    /// A boolean.
    Bool,
    /// A string.
    Str,
    /// A nested dictionary.
    Dict,
    /// Any kind.
    Any,
}

/// What an operation looks like when it is well formed.
#[derive(Clone, Debug)]
pub struct OpSignature {
    /// The operation this describes.
    pub name: OpName,
    /// One line on what the operation means.
    pub summary: &'static str,
    /// How many operands it takes.
    pub operands: Arity,
    /// How many results it produces.
    pub results: Arity,
    /// How many regions it owns.
    pub regions: Arity,
    /// Attributes that must be present, with the kind each must have.
    pub required_attrs: &'static [(&'static str, AttrKind)],
    /// The effect classes the operation may declare. Empty means "any".
    ///
    /// This is the structural half of the safety argument: `agent.action` may
    /// declare anything, but `core.constant` may only be `#pure`, and no
    /// program can relabel a `tool.call` into something a §5 pass would treat
    /// as replayable.
    pub allowed_effects: &'static [EffectClass],
    /// Whether the operation needs its string literal, e.g. the tool name.
    pub requires_literal: bool,
    /// Whether the operation ends its block.
    pub terminator: bool,
}

const PURE_ONLY: &[EffectClass] = &[EffectClass::Pure];
const ANY_EFFECT: &[EffectClass] = &[];
const EXTERNAL: &[EffectClass] = &[
    EffectClass::Pure,
    EffectClass::ReadExternal,
    EffectClass::WriteExternal,
    EffectClass::Irreversible,
];
const NO_ATTRS: &[(&str, AttrKind)] = &[];

/// The registry of every operation the v0.1 dialects define.
#[derive(Clone, Debug)]
pub struct Registry {
    ops: BTreeMap<OpName, OpSignature>,
}

impl Registry {
    /// The v0.1 registry: the six dialects of §3.2.
    pub fn v0_1() -> Self {
        let mut ops = BTreeMap::new();
        for signature in signatures() {
            ops.insert(signature.name.clone(), signature);
        }
        Registry { ops }
    }

    /// The signature of an operation, if the registry knows it.
    pub fn get(&self, name: &OpName) -> Option<&OpSignature> {
        self.ops.get(name)
    }

    /// Whether the registry knows the operation.
    pub fn contains(&self, name: &OpName) -> bool {
        self.ops.contains_key(name)
    }

    /// Every known signature, in name order.
    pub fn iter(&self) -> impl Iterator<Item = &OpSignature> {
        self.ops.values()
    }

    /// Every known signature in one dialect.
    pub fn dialect<'a>(&'a self, dialect: &'a str) -> impl Iterator<Item = &'a OpSignature> {
        self.ops.values().filter(move |s| s.name.dialect == dialect)
    }

    /// How many operations are registered.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::v0_1()
    }
}

fn op(
    dialect: &str,
    name: &str,
    summary: &'static str,
    operands: Arity,
    results: Arity,
    regions: Arity,
    required_attrs: &'static [(&'static str, AttrKind)],
    allowed_effects: &'static [EffectClass],
    requires_literal: bool,
    terminator: bool,
) -> OpSignature {
    OpSignature {
        name: OpName::new(dialect, name),
        summary,
        operands,
        results,
        regions,
        required_attrs,
        allowed_effects,
        requires_literal,
        terminator,
    }
}

fn signatures() -> Vec<OpSignature> {
    use names::*;
    use Arity::*;

    vec![
        // ------------------------------------------------------------- core
        op(CORE, "constant", "A compile-time constant value.",
           Exact(0), Exact(1), Exact(0), &[("value", AttrKind::Any)], PURE_ONLY, false, false),
        op(CORE, "cast", "Reinterprets a value at another type.",
           Exact(1), Exact(1), Exact(0), NO_ATTRS, PURE_ONLY, false, false),
        op(CORE, "cmp", "Compares two values; the literal is the predicate.",
           Exact(2), Exact(1), Exact(0), NO_ATTRS, PURE_ONLY, true, false),

        // ------------------------------------------------------------ agent
        op(AGENT, "func", "An agent program entry point; the literal is its name.",
           Exact(0), Exact(0), Exact(1), NO_ATTRS, PURE_ONLY, true, false),
        op(AGENT, "input", "A value supplied from outside the program.",
           Exact(0), Exact(1), Exact(0), NO_ATTRS, PURE_ONLY, false, false),
        op(AGENT, "context", "The working context: objective and constraints.",
           Any, Exact(1), Exact(0), NO_ATTRS, PURE_ONLY, false, false),
        op(AGENT, "action", "An action the agent takes; the literal names it.",
           Any, Any, Exact(0), NO_ATTRS, ANY_EFFECT, true, false),
        op(AGENT, "plan", "A plan proposed by the LLM builder.",
           Any, Exact(1), Between(0, 1), NO_ATTRS, &[EffectClass::Stochastic], false, false),
        op(AGENT, "budget", "Declares the execution budget of §9.1.",
           Exact(0), Exact(0), Exact(0), NO_ATTRS, PURE_ONLY, false, false),
        op(AGENT, "verify", "Checks a capability before an effectful action (I2).",
           Any, Exact(0), Exact(0), &[("capability", AttrKind::Str)], PURE_ONLY, false, false),
        op(AGENT, "reject", "Discards a candidate value.",
           Exact(1), Exact(0), Exact(0), NO_ATTRS, PURE_ONLY, false, false),
        op(AGENT, "return", "Returns from the enclosing function.",
           Any, Exact(0), Exact(0), NO_ATTRS, PURE_ONLY, false, true),

        // ---------------------------------------------------------- control
        op(CONTROL, "if", "Executes its region when the operand holds.",
           Exact(1), Any, Between(1, 2), NO_ATTRS, PURE_ONLY, false, false),
        op(CONTROL, "while", "Repeats its body while the condition region yields true.",
           Any, Any, Exact(2), &[("max_iterations", AttrKind::Int)], PURE_ONLY, false, false),
        op(CONTROL, "loop", "Iterates its body over the elements of the operand.",
           Exact(1), Any, Exact(1), &[("max_iterations", AttrKind::Int)], PURE_ONLY, false, false),
        op(CONTROL, "parallel", "Executes its region's operations concurrently (I3).",
           Any, Any, Exact(1), NO_ATTRS, PURE_ONLY, false, false),
        op(CONTROL, "yield", "Yields values out of the enclosing region.",
           Any, Exact(0), Exact(0), NO_ATTRS, PURE_ONLY, false, true),

        // ------------------------------------------------------------- tool
        op(TOOL, "call", "Invokes a tool; the literal names it.",
           Any, Any, Exact(0), NO_ATTRS, EXTERNAL, true, false),
        op(TOOL, "result", "Projects one field out of a tool result.",
           Exact(1), Exact(1), Exact(0), &[("field", AttrKind::Str)], PURE_ONLY, false, false),
        op(TOOL, "capability", "Materializes a held capability as a value.",
           Exact(0), Exact(1), Exact(0), &[("capability", AttrKind::Str)], PURE_ONLY, false, false),

        // ----------------------------------------------------------- memory
        op(MEMORY, "read", "Reads an entry from the memory store.",
           Any, Exact(1), Exact(0), &[("key", AttrKind::Str)], &[EffectClass::ReadExternal], false, false),
        op(MEMORY, "write", "Writes an entry to the memory store.",
           Exact(1), Exact(0), Exact(0), &[("key", AttrKind::Str)], &[EffectClass::WriteExternal], false, false),
        op(MEMORY, "search", "Searches memory for entries relevant to the operand.",
           Any, Exact(1), Exact(0), NO_ATTRS, &[EffectClass::ReadExternal], false, false),

        // ------------------------------------------------------ observation
        op(OBSERVATION, "create", "Turns a raw result into a typed observation.",
           Exact(1), Exact(1), Exact(0), NO_ATTRS, PURE_ONLY, false, false),
        op(OBSERVATION, "metric", "Extracts a named metric from an observation.",
           Exact(1), Exact(1), Exact(0), NO_ATTRS, PURE_ONLY, true, false),
        op(OBSERVATION, "error", "Records that an operation failed.",
           Any, Exact(0), Exact(0), &[("message", AttrKind::Str)], PURE_ONLY, false, false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_covers_every_dialect() {
        let registry = Registry::v0_1();
        for dialect in names::ALL {
            assert!(
                registry.dialect(dialect).next().is_some(),
                "dialect `{dialect}` has no operations"
            );
        }
    }

    #[test]
    fn a_tool_call_may_not_be_declared_stochastic() {
        let registry = Registry::v0_1();
        let call = registry.get(&OpName::new("tool", "call")).unwrap();
        assert!(!call.allowed_effects.contains(&EffectClass::Stochastic));
        assert!(call.allowed_effects.contains(&EffectClass::Irreversible));
    }

    #[test]
    fn a_constant_may_only_be_pure() {
        let registry = Registry::v0_1();
        let constant = registry.get(&OpName::new("core", "constant")).unwrap();
        assert_eq!(constant.allowed_effects, PURE_ONLY);
    }

    #[test]
    fn a_loop_must_declare_its_iteration_bound() {
        let registry = Registry::v0_1();
        let looping = registry.get(&OpName::new("control", "loop")).unwrap();
        assert!(looping
            .required_attrs
            .iter()
            .any(|(name, _)| *name == "max_iterations"));
    }

    #[test]
    fn arity_bounds_behave() {
        assert!(Arity::Exact(2).accepts(2));
        assert!(!Arity::Exact(2).accepts(3));
        assert!(Arity::AtLeast(1).accepts(9));
        assert!(!Arity::AtLeast(1).accepts(0));
        assert!(Arity::Between(1, 2).accepts(2));
        assert!(!Arity::Between(1, 2).accepts(3));
        assert!(Arity::Any.accepts(0));
    }
}
