//! # Agent IR analyses
//!
//! The facts every pass and the verifier need, computed once and shared:
//! who uses what, what dominates what, what each operation does to the world,
//! and which operations genuinely have to stay in order.
//!
//! Nothing here mutates the IR. A pass asks these structures a question, and
//! the answer is what its §5 validity condition is allowed to depend on — the
//! point of the exercise being that a pass never has to guess.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod dependency;
pub mod dominance;
pub mod effects;
pub mod uses;

pub use dependency::DependencyGraph;
pub use dominance::{dominates, enclosing_chain, Frame};
pub use effects::EffectSummary;
pub use uses::{Use, UseMap};
