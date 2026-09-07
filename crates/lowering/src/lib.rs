//! # Agent IR lowering and scheduling
//!
//! The last two stages before a runtime sees anything: the `AgentOp → RuntimeOp`
//! table of §6.4, and the scheduler of §15 that orders it, batches what can run
//! together, prices it against the cost model of §9.1, and degrades the
//! strategy when the budget cannot be met.
//!
//! The compiler stops here. It knows a stochastic action is a model call and an
//! external one is a tool call; *which* model and *which* tool runtime is a
//! [`Backend`]'s business, and porting Agent IR to another agent framework
//! means writing one of those and nothing else.
//!
//! ```
//! use agent_ir_lowering::{GenericRuntime, Scheduler};
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
//! assert_eq!(plan.estimated.tool_calls, 1);
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod cost;
pub mod plan;
pub mod scheduler;

pub use cost::{Budget, Cost, CostModel, Overrun};
pub use plan::{
    Backend, Builtin, Control, ExecutionPlan, GenericRuntime, MemoryOp, Plan, RuntimeTarget, Step,
};
pub use scheduler::{estimate, Degradation, Estimator, Scheduler};
