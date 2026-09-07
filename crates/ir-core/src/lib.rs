//! # Agent IR — core intermediate representation
//!
//! This crate implements §2 and §3 of the [specification]: the value types,
//! the effect and capability system, and the
//! `Module → Region → Block → Operation → Value` hierarchy, together with the
//! canonical textual form the parser round-trips.
//!
//! The crate deliberately knows nothing about dialects, passes or runtimes. It
//! answers one question — *what is a well-formed agent program made of* — and
//! leaves *which operations exist* to [`agent-ir-dialects`], *whether the
//! program is legal* to [`agent-ir-verifier`], and *what it means to run it* to
//! [`agent-ir-runtime`].
//!
//! ```
//! use agent_ir_core::{Builder, Effect, Module, Type};
//!
//! let mut module = Module::new("demo");
//! let mut builder = Builder::new(&mut module);
//! builder.func("inspect", [("model", Type::reference("model"))], |b, args| {
//!     let info = b
//!         .op("agent.action")
//!         .literal("inspect_model")
//!         .operand(args[0])
//!         .result("info", Type::observation("model"))
//!         .effect(Effect::Pure)
//!         .build_one();
//!     b.op("agent.return").operand(info).build();
//! });
//!
//! assert!(module.to_string().contains("agent.action \"inspect_model\""));
//! ```
//!
//! [specification]: https://rustnew.github.io/Agent-IR/
//! [`agent-ir-dialects`]: https://docs.rs/agent-ir-dialects
//! [`agent-ir-verifier`]: https://docs.rs/agent-ir-verifier
//! [`agent-ir-runtime`]: https://docs.rs/agent-ir-runtime

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod attribute;
pub mod builder;
pub mod capability;
pub mod diagnostic;
pub mod effect;
pub mod ids;
pub mod module;
pub mod printer;
pub mod provenance;
pub mod types;

pub use attribute::{Attribute, Attributes};
pub use builder::{Builder, OpBuilder};
pub use capability::{Capability, CapabilitySet};
pub use diagnostic::{Diagnostic, Diagnostics, Severity};
pub use effect::{Effect, EffectClass, Scope};
pub use ids::{BlockId, OperationId, RegionId, ValueId};
pub use module::{Block, CacheKey, Module, OpName, Operation, Region, Value, ValueDef};
pub use printer::{print_module, print_operation};
pub use provenance::{Provenance, Source, Validity};
pub use types::{DType, Type};
