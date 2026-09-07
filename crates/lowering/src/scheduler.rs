//! The scheduler of §15.
//!
//! Takes the optimized module, orders it, batches what may run together,
//! estimates the cost, and degrades the strategy when the budget cannot be met.
//!
//! ```text
//! optimized plan → topological sort → batch independent nodes
//!                → estimate cost → budget sufficient? → execution plan
//!                                                     ↘ degraded strategy ↺
//! ```
//!
//! **A note on rigor**, matching the one in §15: this is a greedy heuristic,
//! not an optimizer. Scheduling under multiple constraints is a combinatorial
//! problem, and a real solver is a research question of its own. Batching
//! independent nodes in topological order and degrading on overrun is enough to
//! close the loop, and it is honest about being enough rather than optimal.
//!
//! Unlike the *Parallelization* pass, the scheduler may batch operations that
//! are not adjacent. It is allowed to, because it does not rewrite the program:
//! every dependency edge runs from a lower batch to a higher one, so batch
//! order is a valid topological order and nothing observable moves.

use crate::cost::{Budget, Cost, CostModel, Overrun};
use crate::plan::{Backend, Builtin, Control, ExecutionPlan, Plan, RuntimeTarget, Step};
use agent_ir_analysis::{DependencyGraph, EffectSummary};
use agent_ir_core::{BlockId, Diagnostic, Diagnostics, Module, OperationId};

/// How much one step is assumed to cost before anything has been measured.
///
/// Every field can be overridden per operation with `est_*` attributes, which
/// is how a benchmark run under §9.4 feeds real numbers back into the IR
/// instead of leaving them hard-coded here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimator {
    /// Assumed cost of one tool call.
    pub tool_call: Cost,
    /// Assumed cost of one model call.
    pub inference: Cost,
    /// Assumed cost of one memory operation.
    pub memory: Cost,
    /// Assumed cost of an operation the runtime evaluates itself.
    pub builtin: Cost,
}

impl Default for Estimator {
    fn default() -> Self {
        Estimator {
            tool_call: Cost { tool_calls: 1, latency_ms: 250, ..Cost::ZERO },
            inference: Cost {
                llm_calls: 1,
                input_tokens: 1_000,
                output_tokens: 250,
                latency_ms: 2_000,
                ..Cost::ZERO
            },
            memory: Cost { tool_calls: 1, latency_ms: 20, ..Cost::ZERO },
            builtin: Cost { latency_ms: 1, ..Cost::ZERO },
        }
    }
}

impl Estimator {
    /// The assumed cost for a target, before per-operation overrides.
    pub fn for_target(&self, target: &RuntimeTarget) -> Cost {
        match target {
            RuntimeTarget::ToolCall { .. } => self.tool_call,
            RuntimeTarget::Inference { .. } => self.inference,
            RuntimeTarget::Memory { .. } => self.memory,
            RuntimeTarget::Builtin(_) => self.builtin,
        }
    }

    /// The cost of one operation, with any `est_*` attribute applied.
    pub fn for_operation(
        &self,
        module: &Module,
        op: OperationId,
        target: &RuntimeTarget,
    ) -> Cost {
        let mut cost = self.for_target(target);
        let operation = module.op(op);
        let read = |key: &str| operation.int_attr(key).and_then(|v| u64::try_from(v).ok());
        if let Some(v) = read("est_latency_ms") {
            cost.latency_ms = v;
        }
        if let Some(v) = read("est_input_tokens") {
            cost.input_tokens = v;
        }
        if let Some(v) = read("est_output_tokens") {
            cost.output_tokens = v;
        }
        cost
    }
}

/// How the scheduler responded to a budget it could not meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Degradation {
    /// The budget was met without changing anything.
    None,
    /// Loop bounds were reduced so the plan fits.
    ReducedIterations,
    /// Nothing left to reduce; a human has to decide.
    HumanReview,
}

/// Orders and batches a verified module into an [`ExecutionPlan`].
pub struct Scheduler<B: Backend> {
    backend: B,
    cost_model: CostModel,
    estimator: Estimator,
    budget: Budget,
    max_degradations: usize,
}

impl<B: Backend> Scheduler<B> {
    /// A scheduler for one backend, with no budget constraint.
    pub fn new(backend: B) -> Self {
        Scheduler {
            backend,
            cost_model: CostModel::PLACEHOLDER,
            estimator: Estimator::default(),
            budget: Budget::unlimited(),
            max_degradations: 6,
        }
    }

    /// Constrains the schedule to a budget (§9.1).
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Replaces the cost coefficients.
    pub fn with_cost_model(mut self, model: CostModel) -> Self {
        self.cost_model = model;
        self
    }

    /// Replaces the per-target cost assumptions.
    pub fn with_estimator(mut self, estimator: Estimator) -> Self {
        self.estimator = estimator;
        self
    }

    /// The cost model in use.
    pub fn cost_model(&self) -> CostModel {
        self.cost_model
    }

    /// Schedules one function.
    ///
    /// Fails only when an operation has no lowering; a budget that cannot be
    /// met degrades the strategy and reports it rather than refusing.
    pub fn schedule(
        &self,
        module: &Module,
        function: &str,
    ) -> Result<ExecutionPlan, Diagnostics> {
        let mut errors = Diagnostics::new();

        let Some(func) = module.function(function) else {
            errors.push(
                Diagnostic::error("schedule", format!("no `agent.func \"{function}\"` in this module"))
                    .suggest("check the function name, or list the module's functions first"),
            );
            return Err(errors);
        };

        let region = module.op(func).regions[0];
        let entry = module.region(region).blocks[0];
        let parameters = module.block(entry).args.clone();
        let effects = EffectSummary::build(module);

        let budget = self.declared_budget(module, entry).unwrap_or(self.budget);

        let mut plan = self.schedule_block(module, entry, &effects, &mut errors);
        if errors.has_errors() {
            return Err(errors);
        }

        let mut notes = Vec::new();
        let mut estimated = estimate(&plan);
        let mut degradation = Degradation::None;

        for _ in 0..self.max_degradations {
            let Some(overrun) = budget.overrun(&estimated) else { break };
            if reduce_iterations(&mut plan) {
                degradation = Degradation::ReducedIterations;
                estimated = estimate(&plan);
                notes.push(Diagnostic::warning(
                    "degraded",
                    format!(
                        "the {overrun} was exceeded; loop bounds were halved and the plan \
                         re-estimated"
                    ),
                ));
            } else {
                degradation = Degradation::HumanReview;
                break;
            }
        }

        if let Some(overrun) = budget.overrun(&estimated) {
            degradation = Degradation::HumanReview;
            notes.push(
                Diagnostic::error(
                    "budget",
                    format!(
                        "the {overrun} cannot be met: the plan still costs {estimated} after \
                         degrading"
                    ),
                )
                .suggest(
                    "raise the budget, simplify the program, or route the decision through \
                     HUMAN_REVIEW",
                ),
            );
        }
        let _ = degradation;

        Ok(ExecutionPlan {
            module: module.name.clone(),
            version: module.version,
            function: function.to_string(),
            backend: self.backend.name().to_string(),
            parameters,
            plan,
            estimated,
            notes,
        })
    }

    /// The budget an `agent.budget` operation declares in the entry block.
    fn declared_budget(&self, module: &Module, entry: BlockId) -> Option<Budget> {
        module
            .block(entry)
            .ops
            .iter()
            .copied()
            .filter(|&op| !module.op(op).erased)
            .find(|&op| module.op(op).name.is("agent", "budget"))
            .map(|op| Budget::from_attributes(module.op(op)))
    }

    fn schedule_block(
        &self,
        module: &Module,
        block: BlockId,
        effects: &EffectSummary,
        errors: &mut Diagnostics,
    ) -> Plan {
        let graph = DependencyGraph::of_block(module, block, effects);

        // A terminator always runs last, on its own, whatever level it landed
        // in: batching it beside another operation would let that operation run
        // after the function has returned.
        let (terminators, body): (Vec<OperationId>, Vec<OperationId>) = graph
            .operations()
            .iter()
            .copied()
            .partition(|&op| is_terminator(module, op));

        let mut batches: Vec<Vec<Step>> = Vec::new();
        for level in graph.levels() {
            let batch: Vec<Step> = level
                .into_iter()
                .filter(|op| body.contains(op))
                .filter_map(|op| self.lower_step(module, op, effects, errors))
                .collect();
            if !batch.is_empty() {
                batches.push(batch);
            }
        }
        for op in terminators {
            if let Some(step) = self.lower_step(module, op, effects, errors) {
                batches.push(vec![step]);
            }
        }

        Plan { batches }
    }

    fn lower_step(
        &self,
        module: &Module,
        op: OperationId,
        effects: &EffectSummary,
        errors: &mut Diagnostics,
    ) -> Option<Step> {
        let operation = module.op(op);

        let target = match self.backend.lower(module, op) {
            Ok(target) => target,
            Err(diagnostic) => {
                errors.push(diagnostic);
                return None;
            }
        };

        let control = self.lower_control(module, op, effects, errors);

        // §8.3: anything that is not safely repeatable carries a key the tool
        // runtime checks before re-executing it after a crash.
        let idempotency_key = (!operation.effect.is_replayable()).then(|| {
            format!(
                "{}:{}:{}",
                module.name,
                module.version,
                operation.id.number()
            )
        });

        let mut estimated = self.estimator.for_operation(module, op, &target);
        if let Some(control) = control.as_deref() {
            estimated = control_cost(control, estimated);
        }

        Some(Step {
            op,
            name: operation.name.to_string(),
            literal: operation.literal.clone(),
            target,
            operands: operation.operands.clone(),
            results: operation.results.clone(),
            effect: operation.effect.clone(),
            idempotency_key,
            estimated,
            control,
        })
    }

    fn lower_control(
        &self,
        module: &Module,
        op: OperationId,
        effects: &EffectSummary,
        errors: &mut Diagnostics,
    ) -> Option<Box<Control>> {
        let operation = module.op(op);
        let regions = operation.regions.clone();
        let plan_of = |region: agent_ir_core::RegionId, errors: &mut Diagnostics| {
            let block = module.region(region).blocks[0];
            self.schedule_block(module, block, effects, errors)
        };

        let control = match (operation.name.dialect.as_str(), operation.name.name.as_str()) {
            ("control", "if") => Control::If {
                then: plan_of(regions[0], errors),
                otherwise: regions.get(1).map(|&r| plan_of(r, errors)),
            },
            ("control", "loop") => {
                let block = module.region(regions[0]).blocks[0];
                Control::Loop {
                    binding: module.block(block).args.first().copied(),
                    max_iterations: operation.int_attr("max_iterations").unwrap_or(1),
                    body: plan_of(regions[0], errors),
                }
            }
            ("control", "while") => Control::While {
                condition: plan_of(regions[0], errors),
                max_iterations: operation.int_attr("max_iterations").unwrap_or(1),
                body: plan_of(regions[1], errors),
            },
            ("control", "parallel") => Control::Parallel { body: plan_of(regions[0], errors) },
            _ => return None,
        };
        Some(Box::new(control))
    }
}

fn is_terminator(module: &Module, op: OperationId) -> bool {
    let name = &module.op(op).name;
    name.is("agent", "return") || name.is("control", "yield")
}

/// The cost of a whole plan: batches run in sequence, steps inside a batch run
/// alongside each other.
pub fn estimate(plan: &Plan) -> Cost {
    let mut total = Cost::ZERO;
    for batch in &plan.batches {
        let mut concurrent = Cost::ZERO;
        for step in batch {
            concurrent = concurrent.alongside(step.estimated);
        }
        total = total.then(concurrent);
    }
    total
}

/// Folds a control operation's nested plans into its own cost.
fn control_cost(control: &Control, own: Cost) -> Cost {
    match control {
        Control::If { then, otherwise } => {
            // The worse branch, since which one runs is not known until the
            // condition is evaluated.
            let taken = estimate(then);
            let other = otherwise.as_ref().map(estimate).unwrap_or(Cost::ZERO);
            own.then(if taken.latency_ms >= other.latency_ms { taken } else { other })
        }
        Control::Loop { body, max_iterations, .. } => {
            own.then(repeat(estimate(body), *max_iterations))
        }
        Control::While { condition, body, max_iterations } => {
            own.then(repeat(estimate(condition).then(estimate(body)), *max_iterations))
        }
        Control::Parallel { body } => own.then(estimate(body)),
    }
}

fn repeat(cost: Cost, times: i64) -> Cost {
    let times = times.max(0) as u64;
    Cost {
        input_tokens: cost.input_tokens * times,
        output_tokens: cost.output_tokens * times,
        llm_calls: cost.llm_calls * times,
        tool_calls: cost.tool_calls * times,
        latency_ms: cost.latency_ms * times,
    }
}

/// Halves every loop bound above one, and re-estimates the steps that hold them.
///
/// This is §15's "simplify candidates": fewer benchmark runs rather than none.
/// Returns whether anything could still be reduced.
fn reduce_iterations(plan: &mut Plan) -> bool {
    let mut reduced = false;
    for batch in &mut plan.batches {
        for step in batch {
            let own = step.estimated;
            if let Some(control) = step.control.as_deref_mut() {
                match control {
                    Control::Loop { max_iterations, body, .. }
                    | Control::While { max_iterations, body, .. } => {
                        if *max_iterations > 1 {
                            *max_iterations /= 2;
                            reduced = true;
                        }
                        reduced |= reduce_iterations(body);
                    }
                    Control::If { then, otherwise } => {
                        reduced |= reduce_iterations(then);
                        if let Some(otherwise) = otherwise {
                            reduced |= reduce_iterations(otherwise);
                        }
                    }
                    Control::Parallel { body } => reduced |= reduce_iterations(body),
                }
                // Re-fold the nested cost with the step's own base cost, which
                // the builtin estimate supplies unchanged.
                let base = Cost { latency_ms: own.latency_ms.min(1), ..Cost::ZERO };
                step.estimated = control_cost(step.control.as_deref().unwrap(), base);
            }
        }
    }
    reduced
}

/// Convenience: the first overrun of a plan against a budget.
pub fn overrun_of(plan: &ExecutionPlan, budget: &Budget) -> Option<Overrun> {
    budget.overrun(&plan.estimated)
}

/// The named function's steps, flattened. Handy in tests and in the CLI.
pub fn flatten(plan: &ExecutionPlan) -> Vec<&Step> {
    plan.plan.steps()
}

/// Whether a step reaches outside the program.
pub fn is_external(step: &Step) -> bool {
    !matches!(step.target, RuntimeTarget::Builtin(_))
}

/// Whether a step is the runtime's own evaluation of `builtin`.
pub fn is_builtin(step: &Step, builtin: Builtin) -> bool {
    matches!(&step.target, RuntimeTarget::Builtin(b) if *b == builtin)
}
