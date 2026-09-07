//! Lowering: `AgentOp → RuntimeOp` (§6.4), and the executable plan it produces.
//!
//! The compiler never knows the details of a specific backend. It knows that an
//! `agent.action` whose effect is `#stochastic` is a model call and that one
//! with an external effect is a tool call; which model, and which tool runtime,
//! is a [`Backend`]'s business.
//!
//! ```text
//! agent.action "search"(...)     →  runtime.tool_call(tool = "search")
//! agent.action "run_model"(...)  →  runtime.inference(model = ..., backend = "generic")
//! memory.read {key = "k"}        →  runtime.memory(read, key = "k")
//! core.cmp "lt"(%a, %b)          →  evaluated in the runtime itself
//! ```

use agent_ir_core::{Diagnostic, Effect, Module, OperationId, ValueId};
use serde::{Deserialize, Serialize};
use std::fmt;

/// What the runtime is asked to do for one operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RuntimeTarget {
    /// `runtime.tool_call(tool = ...)`.
    ToolCall {
        /// The tool to invoke.
        tool: String,
    },
    /// `runtime.inference(model = ..., backend = ...)`.
    Inference {
        /// The model, or the action name when the program did not pick one.
        model: String,
        /// The serving backend this lowering targets.
        backend: String,
    },
    /// `runtime.memory(...)`.
    Memory {
        /// Which memory operation.
        operation: MemoryOp,
        /// The key, when the operation names one.
        key: Option<String>,
    },
    /// Evaluated by the runtime itself, with no external dispatch.
    Builtin(Builtin),
}

impl fmt::Display for RuntimeTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeTarget::ToolCall { tool } => write!(f, "runtime.tool_call(tool = \"{tool}\")"),
            RuntimeTarget::Inference { model, backend } => {
                write!(f, "runtime.inference(model = \"{model}\", backend = \"{backend}\")")
            }
            RuntimeTarget::Memory { operation, key } => match key {
                Some(key) => write!(f, "runtime.memory({operation}, key = \"{key}\")"),
                None => write!(f, "runtime.memory({operation})"),
            },
            RuntimeTarget::Builtin(builtin) => write!(f, "runtime.{builtin}"),
        }
    }
}

/// Which memory operation a [`RuntimeTarget::Memory`] performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOp {
    /// Read one entry.
    Read,
    /// Write one entry.
    Write,
    /// Search for relevant entries.
    Search,
}

impl fmt::Display for MemoryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MemoryOp::Read => "read",
            MemoryOp::Write => "write",
            MemoryOp::Search => "search",
        })
    }
}

/// The operations the runtime evaluates on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Builtin {
    /// `core.constant`.
    Constant,
    /// `core.cast`.
    Cast,
    /// `core.cmp`.
    Compare,
    /// `observation.metric`.
    Metric,
    /// `observation.create`.
    Observe,
    /// `observation.error`.
    RecordError,
    /// `tool.result`.
    Project,
    /// `tool.capability`.
    Capability,
    /// `agent.context`.
    Context,
    /// `agent.input`.
    Input,
    /// `agent.budget`.
    Budget,
    /// `agent.verify`.
    Verify,
    /// `agent.reject`.
    Reject,
    /// `agent.return`.
    Return,
    /// `control.yield`.
    Yield,
    /// `control.if` — the runtime drives the nested plans in [`Control::If`].
    If,
    /// `control.loop`.
    Loop,
    /// `control.while`.
    While,
    /// `control.parallel`.
    Parallel,
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Builtin::Constant => "constant",
            Builtin::Cast => "cast",
            Builtin::Compare => "compare",
            Builtin::Metric => "metric",
            Builtin::Observe => "observe",
            Builtin::RecordError => "record_error",
            Builtin::Project => "project",
            Builtin::Capability => "capability",
            Builtin::Context => "context",
            Builtin::Input => "input",
            Builtin::Budget => "budget",
            Builtin::Verify => "verify",
            Builtin::Reject => "reject",
            Builtin::Return => "return",
            Builtin::Yield => "yield",
            Builtin::If => "if",
            Builtin::Loop => "loop",
            Builtin::While => "while",
            Builtin::Parallel => "parallel",
        })
    }
}

/// One lowered operation, ready to execute.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// The IR operation this came from, so a runtime event can point back at
    /// the program (§6, debugging).
    pub op: OperationId,
    /// The operation's name, for traces.
    pub name: String,
    /// The literal, when the operation had one.
    pub literal: Option<String>,
    /// What the runtime should do.
    pub target: RuntimeTarget,
    /// The values consumed.
    pub operands: Vec<ValueId>,
    /// The values produced.
    pub results: Vec<ValueId>,
    /// The declared effect, carried through so the runtime can enforce §8.3.
    pub effect: Effect,
    /// Present for every non-idempotent effect: the key the tool runtime checks
    /// before re-executing after a crash (§8.3).
    pub idempotency_key: Option<String>,
    /// What this step is expected to cost.
    pub estimated: crate::cost::Cost,
    /// Nested structure, for the control operations.
    pub control: Option<Box<Control>>,
}

/// The nested plans a control operation owns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Control {
    /// `control.if`: run `then` when the condition holds, else `otherwise`.
    If {
        /// The branch taken when the condition is true.
        then: Plan,
        /// The branch taken otherwise, when the program supplied one.
        otherwise: Option<Plan>,
    },
    /// `control.loop`: run `body` once per element of the operand.
    Loop {
        /// The block argument each element is bound to.
        binding: Option<ValueId>,
        /// The termination guard of invariant I5.
        max_iterations: i64,
        /// The body.
        body: Plan,
    },
    /// `control.while`: run `body` while `condition` yields true.
    While {
        /// The region computing the condition.
        condition: Plan,
        /// The termination guard of invariant I5.
        max_iterations: i64,
        /// The body.
        body: Plan,
    },
    /// `control.parallel`: run everything in `body` concurrently.
    Parallel {
        /// The body, whose batches are the concurrency the scheduler found.
        body: Plan,
    },
}

/// A scheduled sequence of batches.
///
/// Steps inside one batch have no dependency and no effect conflict, so they
/// may run at the same time; the batches themselves run in order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    /// The batches, in execution order.
    pub batches: Vec<Vec<Step>>,
}

impl Plan {
    /// Every step, flattened into execution order, nested plans included.
    pub fn steps(&self) -> Vec<&Step> {
        let mut out = Vec::new();
        for batch in &self.batches {
            for step in batch {
                out.push(step);
                match step.control.as_deref() {
                    Some(Control::If { then, otherwise }) => {
                        out.extend(then.steps());
                        if let Some(otherwise) = otherwise {
                            out.extend(otherwise.steps());
                        }
                    }
                    Some(Control::Loop { body, .. }) | Some(Control::Parallel { body }) => {
                        out.extend(body.steps())
                    }
                    Some(Control::While { condition, body, .. }) => {
                        out.extend(condition.steps());
                        out.extend(body.steps());
                    }
                    None => {}
                }
            }
        }
        out
    }

    /// How many steps the plan holds at this level.
    pub fn len(&self) -> usize {
        self.batches.iter().map(Vec::len).sum()
    }

    /// Whether the plan does nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The widest batch: the most concurrency the schedule asks for.
    pub fn max_width(&self) -> usize {
        self.batches.iter().map(Vec::len).max().unwrap_or(0)
    }
}

/// A lowered program, ready for a runtime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// The module this came from.
    pub module: String,
    /// The IR version (§1.2).
    pub version: u64,
    /// The function that was lowered.
    pub function: String,
    /// The backend the lowering targeted.
    pub backend: String,
    /// The function's parameters, in order.
    pub parameters: Vec<ValueId>,
    /// The scheduled body.
    pub plan: Plan,
    /// The whole plan's estimated cost, with parallel batches counted as
    /// concurrent.
    pub estimated: crate::cost::Cost,
    /// Anything the scheduler wants to report: a degraded strategy, a budget
    /// that could not be met.
    pub notes: Vec<Diagnostic>,
}

/// Maps IR operations onto runtime operations.
///
/// Implementing this is what porting Agent IR to another agent framework means:
/// the analyses, the passes and the verifier are untouched, only this table
/// changes (§6.4, §12 phase 10).
pub trait Backend {
    /// The backend's name, recorded in the plan and in the event log.
    fn name(&self) -> &'static str;

    /// Lowers one operation, or explains why it cannot.
    fn lower(&self, module: &Module, op: OperationId) -> Result<RuntimeTarget, Diagnostic>;
}

/// The reference backend: dispatches by effect class, with no framework
/// assumptions beyond the dialects themselves.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenericRuntime;

impl Backend for GenericRuntime {
    fn name(&self) -> &'static str {
        "generic"
    }

    fn lower(&self, module: &Module, op: OperationId) -> Result<RuntimeTarget, Diagnostic> {
        let operation = module.op(op);
        let name = &operation.name;

        let builtin = match (name.dialect.as_str(), name.name.as_str()) {
            ("core", "constant") => Some(Builtin::Constant),
            ("core", "cast") => Some(Builtin::Cast),
            ("core", "cmp") => Some(Builtin::Compare),
            ("observation", "metric") => Some(Builtin::Metric),
            ("observation", "create") => Some(Builtin::Observe),
            ("observation", "error") => Some(Builtin::RecordError),
            ("tool", "result") => Some(Builtin::Project),
            ("tool", "capability") => Some(Builtin::Capability),
            ("agent", "context") => Some(Builtin::Context),
            ("agent", "input") => Some(Builtin::Input),
            ("agent", "budget") => Some(Builtin::Budget),
            ("agent", "verify") => Some(Builtin::Verify),
            ("agent", "reject") => Some(Builtin::Reject),
            ("agent", "return") => Some(Builtin::Return),
            ("control", "yield") => Some(Builtin::Yield),
            // Control flow is structure, not dispatch: the runtime walks the
            // nested plans the scheduler attached to the step.
            ("control", "if") => Some(Builtin::If),
            ("control", "loop") => Some(Builtin::Loop),
            ("control", "while") => Some(Builtin::While),
            ("control", "parallel") => Some(Builtin::Parallel),
            _ => None,
        };
        if let Some(builtin) = builtin {
            return Ok(RuntimeTarget::Builtin(builtin));
        }

        match (name.dialect.as_str(), name.name.as_str()) {
            ("memory", "read") => Ok(RuntimeTarget::Memory {
                operation: MemoryOp::Read,
                key: operation.str_attr("key").map(ToString::to_string),
            }),
            ("memory", "write") => Ok(RuntimeTarget::Memory {
                operation: MemoryOp::Write,
                key: operation.str_attr("key").map(ToString::to_string),
            }),
            ("memory", "search") => Ok(RuntimeTarget::Memory {
                operation: MemoryOp::Search,
                key: operation.str_attr("key").map(ToString::to_string),
            }),
            ("agent", "plan") => Ok(RuntimeTarget::Inference {
                model: operation
                    .str_attr("model")
                    .unwrap_or("default")
                    .to_string(),
                backend: self.name().to_string(),
            }),
            // The §6.4 rule: a stochastic action is a model call, an effectful
            // one is a tool call.
            ("agent", "action") | ("tool", "call") => {
                let literal = operation.literal.clone().unwrap_or_default();
                if matches!(operation.effect, Effect::Stochastic) {
                    Ok(RuntimeTarget::Inference {
                        model: operation.str_attr("model").unwrap_or(&literal).to_string(),
                        backend: self.name().to_string(),
                    })
                } else {
                    Ok(RuntimeTarget::ToolCall {
                        tool: operation.str_attr("tool").unwrap_or(&literal).to_string(),
                    })
                }
            }
            _ => Err(Diagnostic::error(
                "lowering",
                format!("the generic backend has no lowering for `{name}`"),
            )
            .at(op)
            .suggest("implement `Backend::lower` for this operation, or remove it before lowering")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_parser::parse_module;

    fn lower_literal(source: &str, literal: &str) -> RuntimeTarget {
        let module = parse_module(source).unwrap();
        let mut found = None;
        module.walk(|op| {
            if op.literal.as_deref() == Some(literal) {
                found = Some(op.id);
            }
        });
        GenericRuntime
            .lower(&module, found.expect("operation not found"))
            .expect("should lower")
    }

    const PROGRAM: &str = r#"module @m version(0) {
  capability @llm scope(*) grants(stochastic)
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %page = agent.action "search"(%url) {effect = #read_external<web>} : !tool.result<page>
    %idea = agent.action "run_model"(%url) {effect = #stochastic, model = "vllm-7b"} : !core.string
    %ok = core.cmp "eq"(%url, %idea) {effect = #pure} : !core.bool
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#;

    #[test]
    fn an_external_action_lowers_to_a_tool_call() {
        assert_eq!(
            lower_literal(PROGRAM, "search"),
            RuntimeTarget::ToolCall { tool: "search".into() }
        );
    }

    #[test]
    fn a_stochastic_action_lowers_to_inference() {
        assert_eq!(
            lower_literal(PROGRAM, "run_model"),
            RuntimeTarget::Inference { model: "vllm-7b".into(), backend: "generic".into() }
        );
    }

    #[test]
    fn a_pure_computation_stays_inside_the_runtime() {
        assert_eq!(
            lower_literal(PROGRAM, "eq"),
            RuntimeTarget::Builtin(Builtin::Compare)
        );
    }

    #[test]
    fn memory_lowers_with_its_key() {
        let module = parse_module(
            r#"module @m version(0) {
  capability @mem scope("mem") grants(read_external)

  agent.func "f" {
    %v = memory.read {effect = #read_external<mem>, key = "last_run"} : !memory.memory
    agent.return(%v) {effect = #pure}
  } {effect = #pure}
}
"#,
        )
        .unwrap();
        let read = module
            .op_ids()
            .into_iter()
            .find(|&id| module.op(id).name.is("memory", "read"))
            .unwrap();
        assert_eq!(
            GenericRuntime.lower(&module, read).unwrap(),
            RuntimeTarget::Memory { operation: MemoryOp::Read, key: Some("last_run".into()) }
        );
    }

    #[test]
    fn an_unlowerable_operation_reports_rather_than_panics() {
        let module = parse_module(
            r#"module @m version(0) {
  agent.func "f" {
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        )
        .unwrap();
        let func = module.function("f").unwrap();
        let error = GenericRuntime.lower(&module, func).unwrap_err();
        assert_eq!(error.code, "lowering");
        assert!(error.suggestion.is_some());
    }

    #[test]
    fn a_target_prints_the_section_6_4_form() {
        assert_eq!(
            RuntimeTarget::ToolCall { tool: "web_search".into() }.to_string(),
            "runtime.tool_call(tool = \"web_search\")"
        );
        assert_eq!(
            RuntimeTarget::Inference { model: "m".into(), backend: "vllm".into() }.to_string(),
            "runtime.inference(model = \"m\", backend = \"vllm\")"
        );
    }
}
