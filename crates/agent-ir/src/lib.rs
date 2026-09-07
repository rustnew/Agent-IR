//! # Agent IR
//!
//! A compilation infrastructure for agentic systems. This crate is the facade:
//! it re-exports the pieces and wires them into the pipeline of §4.
//!
//! ```text
//! text → parse → verify → analyse → optimize → verify safety → schedule → lower → run
//!                   ↓                                ↓
//!            structured diagnostic            structured diagnostic
//! ```
//!
//! Both rejection points return [`Diagnostics`] rather than an exception, which
//! is what makes an Agent IR program debuggable like a compiler input rather
//! than like a chain of prompts.
//!
//! ```
//! let module = agent_ir::compile(
//!     r#"
//!     module @m version(0) {
//!       capability @web scope("web") grants(read_external)
//!
//!       agent.func "f" {
//!       ^bb0(%url: !core.string):
//!         %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
//!         agent.return(%page) {effect = #pure}
//!       } {effect = #pure}
//!     }
//!     "#,
//! )
//! .unwrap();
//!
//! assert!(module.function("f").is_some());
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub use agent_ir_analysis as analysis;
pub use agent_ir_core as core;
pub use agent_ir_dialects as dialects;
pub use agent_ir_lowering as lowering;
pub use agent_ir_parser as parser;
pub use agent_ir_passes as passes;
pub use agent_ir_runtime as runtime;
pub use agent_ir_verifier as verifier;

use agent_ir_core::{Diagnostic, Diagnostics, Module};

/// Everything that can stop a program before it runs.
#[derive(Debug)]
pub enum Error {
    /// The text was not a well-formed module.
    Parse(agent_ir_parser::ParseError),
    /// The module broke an invariant, or the agent lacked a capability.
    Rejected(Diagnostics),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(err) => write!(f, "{err}"),
            Error::Rejected(report) => write!(f, "{report}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<agent_ir_parser::ParseError> for Error {
    fn from(err: agent_ir_parser::ParseError) -> Self {
        Error::Parse(err)
    }
}

impl Error {
    /// The findings, when the program was rejected rather than malformed.
    pub fn diagnostics(&self) -> Diagnostics {
        match self {
            Error::Rejected(report) => report.clone(),
            Error::Parse(err) => {
                let mut report = Diagnostics::new();
                report.push(
                    Diagnostic::error("parse", err.message.clone()).suggest(format!(
                        "at line {}, column {}",
                        err.span.line, err.span.column
                    )),
                );
                report
            }
        }
    }
}

/// Parses and verifies a module, without optimizing it.
///
/// Both phases of §4 run: a module that comes back is well formed *and*
/// authorized.
pub fn compile(source: &str) -> Result<Module, Error> {
    let module = agent_ir_parser::parse_module(source)?;
    let report = agent_ir_verifier::Verifier::new().verify_all(&module);
    if report.has_errors() {
        return Err(Error::Rejected(report));
    }
    Ok(module)
}

/// Parses, verifies, optimizes, and verifies again.
///
/// The second verification is not ceremony. A pass that broke an invariant
/// would otherwise hand the runtime a program the compiler already promised was
/// safe, and §5 is explicit that no pass is safe by default.
pub fn compile_optimized(source: &str) -> Result<(Module, agent_ir_passes::PipelineReport), Error> {
    let mut module = compile(source)?;
    let report = agent_ir_passes::PassManager::default_pipeline().run(&mut module);
    let after = agent_ir_verifier::Verifier::new().verify_all(&module);
    if after.has_errors() {
        return Err(Error::Rejected(after));
    }
    Ok((module, report))
}

/// The full pipeline: compile, optimize, and schedule one function.
pub fn compile_and_schedule(
    source: &str,
    function: &str,
) -> Result<agent_ir_lowering::ExecutionPlan, Error> {
    let (module, _) = compile_optimized(source)?;
    agent_ir_lowering::Scheduler::new(agent_ir_lowering::GenericRuntime)
        .schedule(&module, function)
        .map_err(Error::Rejected)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#;

    #[test]
    fn the_pipeline_runs_end_to_end() {
        let plan = compile_and_schedule(GOOD, "f").unwrap();
        assert_eq!(plan.function, "f");
        assert_eq!(plan.estimated.tool_calls, 1);
    }

    #[test]
    fn a_malformed_module_reports_where() {
        let error = compile("module @m version(0) { oops }").unwrap_err();
        assert!(matches!(error, Error::Parse(_)));
        assert!(!error.diagnostics().is_empty());
    }

    #[test]
    fn an_unauthorized_module_is_rejected_before_it_can_run() {
        let error = compile(
            r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    agent.verify(%db) {effect = #pure, capability = "wipe"}
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        )
        .unwrap_err();
        let report = error.diagnostics();
        assert!(report.errors().any(|d| d.code == "I2"), "{report}");
    }

    #[test]
    fn optimizing_is_verified_again_afterwards() {
        let (module, report) = compile_optimized(GOOD).unwrap();
        assert!(!report.changed, "nothing to optimize here");
        assert!(module.function("f").is_some());
    }
}
