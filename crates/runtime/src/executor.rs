//! The executor: it runs a lowered plan and nothing else.
//!
//! §6.1 is explicit that the runtime executes the plan and does not reinterpret
//! it. Everything the executor is allowed to decide — batching, ordering,
//! budgets — was already decided by the scheduler; what is left here is
//! evaluating the builtins, driving control flow, dispatching effects through
//! the [`Environment`], and the durability machinery of §8: idempotency keys,
//! loop guards, checkpoints and an append-only log.
//!
//! ## Concurrency
//!
//! A batch is a set of steps the scheduler proved may run at the same time.
//! This reference executor runs them one after another anyway, single-threaded,
//! and says so rather than pretending otherwise: the concurrency lives in the
//! plan, and exploiting it is a backend's job. The latency figures in
//! [`agent_ir_lowering::Cost`] describe what a concurrent runtime would
//! achieve, and §9.4 is where that claim gets checked against a real one.

use crate::env::{EnvError, Environment, Invocation};
use crate::event::{EventKind, EventLog};
use crate::state::{Checkpoint, CheckpointStore, ExecutionState};
use crate::value::Value;
use agent_ir_core::ValueId;
use agent_ir_lowering::{Builtin, Control, ExecutionPlan, Plan, RuntimeTarget, Step};
use std::collections::BTreeMap;
use std::fmt;

/// What the runtime does when a loop guard fires (§8.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopGuardAction {
    /// Pause and try again. The reference executor has no scheduler to pause
    /// on, so it stops and reports; a production runtime would back off.
    RetryWithBackoff,
    /// Give up on this strategy and hand control back to the builder.
    ChangeStrategy,
    /// Stop and ask a person.
    HumanReview,
}

impl fmt::Display for LoopGuardAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LoopGuardAction::RetryWithBackoff => "RETRY_WITH_BACKOFF",
            LoopGuardAction::ChangeStrategy => "CHANGE_STRATEGY",
            LoopGuardAction::HumanReview => "HUMAN_REVIEW",
        })
    }
}

/// The durability knobs of §8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimePolicy {
    /// How many completed effects between checkpoints. Zero disables them.
    pub checkpoint_every: usize,
    /// How many identical repetitions of an action before the guard fires.
    pub repeat_threshold: usize,
    /// What the guard does when it fires.
    pub on_loop_guard: LoopGuardAction,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        RuntimePolicy {
            checkpoint_every: 8,
            repeat_threshold: 16,
            on_loop_guard: LoopGuardAction::HumanReview,
        }
    }
}

/// Why an execution stopped early.
#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeError {
    /// The world failed, or vanished.
    Environment(EnvError),
    /// An action repeated identically past the threshold (§8.4).
    LoopGuard {
        /// The repeated action.
        target: String,
        /// How many times it repeated.
        repeats: usize,
        /// What the policy decided.
        action: LoopGuardAction,
    },
    /// A loop ran past its declared `max_iterations` (invariant I5).
    IterationLimit {
        /// The declared bound.
        limit: i64,
    },
    /// A value was not the shape the operation needed.
    Type(String),
    /// The plan asked for something this executor does not implement.
    Unsupported(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::Environment(err) => write!(f, "{err}"),
            RuntimeError::LoopGuard {
                target,
                repeats,
                action,
            } => write!(
                f,
                "loop guard: `{target}` repeated identically {repeats} times, taking {action}"
            ),
            RuntimeError::IterationLimit { limit } => {
                write!(
                    f,
                    "a loop exceeded its declared bound of {limit} iterations"
                )
            }
            RuntimeError::Type(message) => write!(f, "type error at run time: {message}"),
            RuntimeError::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<EnvError> for RuntimeError {
    fn from(err: EnvError) -> Self {
        RuntimeError::Environment(err)
    }
}

/// What an execution produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// What the function returned.
    pub result: Value,
    /// The state at the end, including the idempotency ledger.
    pub state: ExecutionState,
    /// How many effects actually reached the world.
    pub effects: usize,
    /// How many were skipped because their key was already recorded (§8.3).
    pub replayed: usize,
}

/// Where control went after a step or a plan.
enum Flow {
    /// Keep going.
    Continue,
    /// The function returned.
    Return(Vec<Value>),
    /// A region yielded its results.
    Yield(Vec<Value>),
}

/// Runs a lowered plan against an environment.
pub struct Executor<'a, E: Environment, L: EventLog, C: CheckpointStore> {
    env: &'a mut E,
    log: &'a mut L,
    checkpoints: &'a mut C,
    policy: RuntimePolicy,
    state: ExecutionState,
    version: u64,
    effects: usize,
    replayed: usize,
    since_checkpoint: usize,
    checkpoint_sequence: u64,
    /// The loop iterations currently open, outermost first.
    ///
    /// An idempotency key names an *invocation*, not an operation: the write
    /// inside a forty-iteration loop is forty different effects, and giving
    /// them one key would let recovery skip thirty-nine of them. The static
    /// half of the key comes from the compiler (module, version, operation);
    /// this is the dynamic half.
    ///
    /// It is stable across a replay because the plan is: the loop iterates the
    /// same list in the same order. That holds even when the list came from a
    /// model — a `#stochastic` call is non-replayable, so it carries a key of
    /// its own and is memoized rather than re-sampled.
    path: Vec<usize>,
}

impl<'a, E: Environment, L: EventLog, C: CheckpointStore> Executor<'a, E, L, C> {
    /// An executor with the default policy.
    pub fn new(env: &'a mut E, log: &'a mut L, checkpoints: &'a mut C) -> Self {
        Executor {
            env,
            log,
            checkpoints,
            policy: RuntimePolicy::default(),
            state: ExecutionState::new(),
            version: 0,
            effects: 0,
            replayed: 0,
            since_checkpoint: 0,
            checkpoint_sequence: 0,
            path: Vec::new(),
        }
    }

    /// Replaces the durability policy.
    pub fn with_policy(mut self, policy: RuntimePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Starts from an existing state — the ledger of a previous, crashed run.
    ///
    /// This is what makes recovery work: the plan is re-run from the top, and
    /// every effect whose key is already in the ledger is skipped rather than
    /// repeated. It is sound because the effect system guarantees that the
    /// steps *not* in the ledger are exactly the replayable ones.
    pub fn resuming_from(mut self, state: ExecutionState) -> Self {
        self.state = state;
        self
    }

    /// The state as it stands.
    pub fn state(&self) -> &ExecutionState {
        &self.state
    }

    /// Runs one lowered function.
    pub fn run(
        &mut self,
        plan: &ExecutionPlan,
        arguments: Vec<Value>,
    ) -> Result<Outcome, RuntimeError> {
        self.version = plan.version;
        for (id, value) in plan.parameters.iter().zip(arguments) {
            self.state.bind(*id, value);
        }
        self.log.append(
            None,
            EventKind::Started {
                function: plan.function.clone(),
                version: plan.version,
            },
        );

        let flow = self.run_plan(&plan.plan)?;
        let result = match flow {
            Flow::Return(values) | Flow::Yield(values) => single(values),
            Flow::Continue => Value::Null,
        };

        self.log.append(
            None,
            EventKind::Finished {
                result: result.clone(),
            },
        );
        Ok(Outcome {
            result,
            state: self.state.clone(),
            effects: self.effects,
            replayed: self.replayed,
        })
    }

    fn run_plan(&mut self, plan: &Plan) -> Result<Flow, RuntimeError> {
        for batch in &plan.batches {
            // The scheduler proved these may run together; this executor runs
            // them in order anyway, see the module note on concurrency.
            for step in batch {
                match self.run_step(step)? {
                    Flow::Continue => {}
                    flow => return Ok(flow),
                }
            }
        }
        Ok(Flow::Continue)
    }

    fn run_step(&mut self, step: &Step) -> Result<Flow, RuntimeError> {
        match &step.target {
            RuntimeTarget::Builtin(builtin) => self.run_builtin(step, *builtin),
            _ => {
                self.run_external(step)?;
                Ok(Flow::Continue)
            }
        }
    }

    // ----------------------------------------------------------- the world

    /// The key that identifies this invocation, static half plus loop path.
    fn effective_key(&self, step: &Step) -> Option<String> {
        step.idempotency_key.as_ref().map(|key| {
            let path: Vec<String> = self.path.iter().map(usize::to_string).collect();
            format!("{key}@{}", path.join("."))
        })
    }

    fn run_external(&mut self, step: &Step) -> Result<(), RuntimeError> {
        let mut call = self.invocation(step);
        let key = self.effective_key(step);
        call.idempotency_key = key.clone();

        // §8.3, first question: did this already happen?
        if let Some(key) = &key {
            let recorded = self
                .state
                .ledger
                .get(key)
                .cloned()
                .or_else(|| self.env.lookup_idempotent(key));
            if let Some(result) = recorded {
                self.log.append(
                    Some(step.op),
                    EventKind::EffectReplayed {
                        target: call.target.clone(),
                        idempotency_key: key.clone(),
                        result: result.clone(),
                    },
                );
                self.bind_results(step, result);
                self.replayed += 1;
                return Ok(());
            }
        }

        self.check_loop_guard(step, &call)?;

        let outcome = match &step.target {
            RuntimeTarget::ToolCall { .. } => self.env.tool_call(&call),
            RuntimeTarget::Inference { .. } => self.env.infer(&call),
            RuntimeTarget::Memory { operation, .. } => match operation {
                agent_ir_lowering::MemoryOp::Read => self.env.memory_read(&call),
                agent_ir_lowering::MemoryOp::Write => self.env.memory_write(&call),
                agent_ir_lowering::MemoryOp::Search => self.env.memory_search(&call),
            },
            RuntimeTarget::Builtin(_) => unreachable!("builtins never reach the world"),
        };

        let result = match outcome {
            Ok(result) => result,
            Err(err) => {
                self.log.append(
                    Some(step.op),
                    EventKind::Failed {
                        target: call.target.clone(),
                        error: err.to_string(),
                    },
                );
                return Err(err.into());
            }
        };

        if let Some(key) = &key {
            self.state.ledger.record(key.clone(), result.clone());
        }
        self.log.append(
            Some(step.op),
            EventKind::Effect {
                target: call.target.clone(),
                effect: step.effect.to_string(),
                idempotency_key: key.clone(),
                result: result.clone(),
            },
        );
        self.bind_results(step, result);
        self.effects += 1;
        self.maybe_checkpoint();
        Ok(())
    }

    fn invocation(&self, step: &Step) -> Invocation {
        let target = match &step.target {
            RuntimeTarget::ToolCall { tool } => tool.clone(),
            RuntimeTarget::Inference { model, .. } => model.clone(),
            RuntimeTarget::Memory { key, .. } => key.clone().unwrap_or_default(),
            RuntimeTarget::Builtin(builtin) => builtin.to_string(),
        };
        Invocation {
            target,
            args: step
                .operands
                .iter()
                .map(|&id| self.state.get_or_null(id))
                .collect(),
            attributes: step
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), Value::from(v)))
                .collect(),
            effect: step.effect.clone(),
            idempotency_key: step.idempotency_key.clone(),
        }
    }

    /// §8.4: the same action, with the same arguments, past a threshold.
    fn check_loop_guard(&mut self, step: &Step, call: &Invocation) -> Result<(), RuntimeError> {
        let signature = format!(
            "{}|{}",
            call.target,
            call.args
                .iter()
                .map(Value::fingerprint)
                .collect::<Vec<_>>()
                .join(",")
        );
        let repeats = self.state.count_repeat(signature);
        if repeats > self.policy.repeat_threshold {
            let action = self.policy.on_loop_guard;
            self.log.append(
                Some(step.op),
                EventKind::LoopGuard {
                    target: call.target.clone(),
                    repeats,
                    action: action.to_string(),
                },
            );
            return Err(RuntimeError::LoopGuard {
                target: call.target.clone(),
                repeats,
                action,
            });
        }
        Ok(())
    }

    fn maybe_checkpoint(&mut self) {
        if self.policy.checkpoint_every == 0 {
            return;
        }
        self.since_checkpoint += 1;
        if self.since_checkpoint < self.policy.checkpoint_every {
            return;
        }
        self.since_checkpoint = 0;
        self.checkpoint_sequence += 1;
        // The snapshot is taken whole, then logged: a reader of the log never
        // sees a checkpoint that does not exist in the store.
        self.checkpoints.save(Checkpoint {
            sequence: self.checkpoint_sequence,
            log_position: self.log.len() as u64,
            version: self.version,
            state: self.state.clone(),
        });
        self.log.append(
            None,
            EventKind::Checkpoint {
                sequence: self.checkpoint_sequence,
                effects: self.state.ledger.len(),
            },
        );
    }

    fn bind_results(&mut self, step: &Step, result: Value) {
        match step.results.len() {
            0 => {}
            1 => self.state.bind(step.results[0], result),
            n => match result {
                Value::List(items) if items.len() == n => {
                    for (id, item) in step.results.iter().zip(items) {
                        self.state.bind(*id, item);
                    }
                }
                other => {
                    // Not the shape the operation promised. Binding null is
                    // better than binding a lie; the type mismatch surfaces at
                    // the first read.
                    for &id in &step.results {
                        self.state.bind(id, Value::Null);
                    }
                    let _ = other;
                }
            },
        }
    }

    // -------------------------------------------------------------- builtins

    fn run_builtin(&mut self, step: &Step, builtin: Builtin) -> Result<Flow, RuntimeError> {
        let operand = |index: usize| -> Value {
            step.operands
                .get(index)
                .map(|&id| self.state.get_or_null(id))
                .unwrap_or(Value::Null)
        };

        match builtin {
            Builtin::Constant => {
                let value = step
                    .attributes
                    .get("value")
                    .map(Value::from)
                    .unwrap_or(Value::Null);
                self.bind_results(step, value);
            }
            Builtin::Cast | Builtin::Observe | Builtin::Input => {
                self.bind_results(step, operand(0));
            }
            Builtin::Compare => {
                let value = self.compare(step, operand(0), operand(1))?;
                self.bind_results(step, value);
            }
            Builtin::Metric => {
                let name = step.literal.clone().unwrap_or_default();
                let value = operand(0).field(&name).cloned().ok_or_else(|| {
                    RuntimeError::Type(format!(
                        "`observation.metric \"{name}\"` needs a record with that field, got {}",
                        operand(0)
                    ))
                })?;
                self.bind_results(step, value);
            }
            Builtin::Project => {
                let field = step
                    .attributes
                    .get("field")
                    .and_then(agent_ir_core::Attribute::as_str)
                    .unwrap_or_default()
                    .to_string();
                let value = operand(0).field(&field).cloned().unwrap_or(Value::Null);
                self.bind_results(step, value);
            }
            Builtin::Capability => {
                let name = step
                    .attributes
                    .get("capability")
                    .and_then(agent_ir_core::Attribute::as_str)
                    .unwrap_or_default();
                self.bind_results(step, Value::Str(name.to_string()));
            }
            Builtin::Context => {
                let mut record: BTreeMap<String, Value> = step
                    .attributes
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(v)))
                    .collect();
                for (index, &id) in step.operands.iter().enumerate() {
                    record.insert(format!("operand_{index}"), self.state.get_or_null(id));
                }
                self.bind_results(step, Value::Record(record));
            }
            Builtin::Budget | Builtin::Verify => {
                // Both were settled at compile time. `agent.verify` in
                // particular is not a runtime check: if the program reached the
                // runtime at all, the capability was there (§7).
            }
            Builtin::RecordError => {
                let message = step
                    .attributes
                    .get("message")
                    .and_then(agent_ir_core::Attribute::as_str)
                    .unwrap_or("unspecified")
                    .to_string();
                self.log.append(
                    Some(step.op),
                    EventKind::Failed {
                        target: "observation.error".into(),
                        error: message,
                    },
                );
            }
            Builtin::Reject => {
                let rejected = operand(0);
                self.state.rejected.push(rejected);
            }
            Builtin::Return => {
                let values: Vec<Value> = step
                    .operands
                    .iter()
                    .map(|&id| self.state.get_or_null(id))
                    .collect();
                return Ok(Flow::Return(values));
            }
            Builtin::Yield => {
                let values: Vec<Value> = step
                    .operands
                    .iter()
                    .map(|&id| self.state.get_or_null(id))
                    .collect();
                return Ok(Flow::Yield(values));
            }
            Builtin::If | Builtin::Loop | Builtin::While | Builtin::Parallel => {
                return self.run_control(step);
            }
        }
        Ok(Flow::Continue)
    }

    fn compare(&self, step: &Step, left: Value, right: Value) -> Result<Value, RuntimeError> {
        let predicate = step.literal.as_deref().unwrap_or("eq");
        if matches!(predicate, "eq" | "ne") {
            let equal = left == right;
            return Ok(Value::Bool(if predicate == "eq" { equal } else { !equal }));
        }
        let (a, b) = match (left.as_float(), right.as_float()) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err(RuntimeError::Type(format!(
                    "`core.cmp \"{predicate}\"` needs two numbers, got {left} and {right}"
                )))
            }
        };
        Ok(Value::Bool(match predicate {
            "lt" => a < b,
            "le" => a <= b,
            "gt" => a > b,
            "ge" => a >= b,
            other => {
                return Err(RuntimeError::Unsupported(format!(
                    "comparison predicate `{other}`"
                )))
            }
        }))
    }

    // -------------------------------------------------------- control flow

    fn run_control(&mut self, step: &Step) -> Result<Flow, RuntimeError> {
        let Some(control) = step.control.as_deref() else {
            return Err(RuntimeError::Unsupported(format!(
                "`{}` reached the runtime with no nested plan",
                step.name
            )));
        };

        match control {
            Control::If { then, otherwise } => {
                let condition = step
                    .operands
                    .first()
                    .map(|&id| self.state.get_or_null(id))
                    .unwrap_or(Value::Null);
                let taken = condition.is_true().ok_or_else(|| {
                    RuntimeError::Type(format!(
                        "`control.if` needs a boolean condition, got {condition}"
                    ))
                })?;
                let branch = if taken {
                    Some(then)
                } else {
                    otherwise.as_ref()
                };
                match branch {
                    Some(plan) => self.run_region(step, plan),
                    None => Ok(Flow::Continue),
                }
            }
            Control::Loop {
                binding,
                max_iterations,
                body,
            } => {
                let iterable = step
                    .operands
                    .first()
                    .map(|&id| self.state.get_or_null(id))
                    .unwrap_or(Value::Null);
                let items: Vec<Value> = match iterable {
                    Value::List(items) => items,
                    Value::Null => Vec::new(),
                    other => vec![other],
                };
                if items.len() as i64 > *max_iterations {
                    // Invariant I5 is a promise the program made. Breaking it at
                    // run time is an error, not something to quietly truncate.
                    return Err(RuntimeError::IterationLimit {
                        limit: *max_iterations,
                    });
                }
                for (iteration, item) in items.into_iter().enumerate() {
                    if let Some(id) = binding {
                        self.state.bind(*id, item);
                    }
                    self.path.push(iteration);
                    let flow = self.run_plan(body);
                    self.path.pop();
                    match flow? {
                        Flow::Return(values) => return Ok(Flow::Return(values)),
                        Flow::Continue | Flow::Yield(_) => {}
                    }
                }
                Ok(Flow::Continue)
            }
            Control::While {
                condition,
                max_iterations,
                body,
            } => {
                for iteration in 0..*max_iterations {
                    let holds = match self.run_plan(condition)? {
                        Flow::Yield(values) => single(values).is_true().ok_or_else(|| {
                            RuntimeError::Type(
                                "`control.while` needs its condition region to yield a boolean"
                                    .into(),
                            )
                        })?,
                        Flow::Return(values) => return Ok(Flow::Return(values)),
                        Flow::Continue => false,
                    };
                    if !holds {
                        return Ok(Flow::Continue);
                    }
                    self.path.push(iteration as usize);
                    let flow = self.run_plan(body);
                    self.path.pop();
                    match flow? {
                        Flow::Return(values) => return Ok(Flow::Return(values)),
                        Flow::Continue | Flow::Yield(_) => {}
                    }
                    if iteration + 1 == *max_iterations {
                        return Err(RuntimeError::IterationLimit {
                            limit: *max_iterations,
                        });
                    }
                }
                Ok(Flow::Continue)
            }
            Control::Parallel { body } => self.run_region(step, body),
        }
    }

    /// Runs a region and binds whatever it yields to the owning step's results.
    fn run_region(&mut self, step: &Step, plan: &Plan) -> Result<Flow, RuntimeError> {
        match self.run_plan(plan)? {
            Flow::Return(values) => Ok(Flow::Return(values)),
            Flow::Yield(values) => {
                for (id, value) in step.results.iter().zip(values) {
                    self.state.bind(*id, value);
                }
                Ok(Flow::Continue)
            }
            Flow::Continue => Ok(Flow::Continue),
        }
    }
}

fn single(mut values: Vec<Value>) -> Value {
    match values.len() {
        0 => Value::Null,
        1 => values.remove(0),
        _ => Value::List(values),
    }
}

/// Convenience: the values a step bound, for tests and the CLI.
pub fn bound(state: &ExecutionState, ids: &[ValueId]) -> Vec<Value> {
    ids.iter().map(|&id| state.get_or_null(id)).collect()
}
