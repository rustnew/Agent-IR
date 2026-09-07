//! # Agent IR optimization passes
//!
//! Every pass in this crate states its §5 validity condition in terms of the
//! effect system and refuses to fire when the condition is not met. That is the
//! whole point of the exercise: `Dead Action Elimination` is not "remove
//! operations whose results are unused", it is "remove operations whose results
//! are unused **and** whose effect summary is replayable", and the second half
//! is what stops it deleting a payment.
//!
//! Passes are deliberately conservative in the direction that keeps them sound.
//! Where a transformation would need a proof the IR cannot supply, the pass
//! declines rather than guesses — [`Parallelization`] only fuses operations
//! that are already adjacent, for a reason documented on the pass itself.
//!
//! ```
//! use agent_ir_passes::PassManager;
//!
//! let mut module = agent_ir_parser::parse_module(r#"
//!     module @m version(0) {
//!       capability @web scope("web") grants(read_external)
//!
//!       agent.func "f" {
//!       ^bb0(%url: !core.string):
//!         %unused = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
//!         agent.return {effect = #pure}
//!       } {effect = #pure}
//!     }
//! "#).unwrap();
//!
//! let report = PassManager::default_pipeline().run(&mut module);
//! assert!(report.changed);
//! assert!(!module.to_string().contains("fetch"));
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod dead_action;
pub mod dedup;
pub mod manager;
pub mod parallelize;

pub use dead_action::DeadActionElimination;
pub use dedup::ToolCallDeduplication;
pub use manager::{Pass, PassManager, PassReport, PipelineReport};
pub use parallelize::Parallelization;
