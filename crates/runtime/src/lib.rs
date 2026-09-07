//! # Agent IR runtime
//!
//! Durable execution of a lowered plan: §8 in full, and the worked crash of
//! §16.
//!
//! The runtime executes the plan and does not reinterpret it (§6.1). What it
//! adds is durability — an append-only event log, checkpoints, idempotency keys
//! and loop guards — and the one idea that ties this crate back to the rest of
//! the project: **recovery is replay, and the effect system is what makes
//! replay sound.** A step the compiler proved replayable is simply re-run; a
//! step it proved otherwise is memoized under its idempotency key and never
//! happens twice.
//!
//! ```
//! use agent_ir_lowering::{GenericRuntime, Scheduler};
//! use agent_ir_runtime::{
//!     Executor, InMemoryCheckpointStore, InMemoryEventLog, RecordingEnvironment, Value,
//! };
//!
//! let module = agent_ir_parser::parse_module(r#"
//!     module @m version(0) {
//!       capability @web scope("web") grants(read_external)
//!
//!       agent.func "f" {
//!       ^bb0(%url: !core.string):
//!         %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
//!         agent.return(%page) {effect = #pure}
//!       } {effect = #pure}
//!     }
//! "#).unwrap();
//!
//! let plan = Scheduler::new(GenericRuntime).schedule(&module, "f").unwrap();
//! let mut env = RecordingEnvironment::new().returning("fetch", Value::Str("<html>".into()));
//! let mut log = InMemoryEventLog::new();
//! let mut checkpoints = InMemoryCheckpointStore::new();
//!
//! let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
//!     .run(&plan, vec![Value::Str("https://example.com".into())])
//!     .unwrap();
//!
//! assert_eq!(outcome.result, Value::Str("<html>".into()));
//! assert_eq!(env.call_names(), vec!["fetch"]);
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod env;
pub mod event;
pub mod executor;
pub mod recovery;
pub mod state;
pub mod value;

pub use env::{EnvError, Environment, Invocation, RecordingEnvironment};
pub use event::{Event, EventKind, EventLog, InMemoryEventLog};
pub use executor::{Executor, LoopGuardAction, Outcome, RuntimeError, RuntimePolicy};
pub use recovery::{RecoveryManager, Resumption};
pub use state::{
    Checkpoint, CheckpointStore, ExecutionState, InMemoryCheckpointStore, Ledger,
};
pub use value::Value;
