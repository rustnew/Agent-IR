//! The pass trait and the pass manager.
//!
//! The manager runs its passes to a fixpoint, because one pass usually creates
//! work for another: deduplicating a tool call makes its result unused, which
//! makes the call dead, which frees the operations around it to be scheduled
//! together. It stops when a full round changes nothing, or at an iteration
//! cap so a mis-written pass cannot spin forever.

use agent_ir_core::{Diagnostics, Module, OperationId, ValueId};
use std::fmt;

/// One transformation over a module.
///
/// A pass reports what it did rather than mutating silently, because §9.3 lists
/// "optimization overhead" as a metric that has to stay well under the gain,
/// and you cannot measure what you do not record.
pub trait Pass {
    /// The pass name, as it appears in reports and on the command line.
    fn name(&self) -> &'static str;

    /// One line on what the pass does and what makes it legal.
    fn description(&self) -> &'static str;

    /// Applies the pass, returning what changed.
    fn run(&self, module: &mut Module) -> PassReport;
}

/// What a single pass did to a module.
#[derive(Clone, Debug, Default)]
pub struct PassReport {
    /// Which pass produced this report.
    pub pass: String,
    /// Whether the module changed at all.
    pub changed: bool,
    /// Operations the pass removed.
    pub removed: Vec<OperationId>,
    /// Operations the pass introduced.
    pub created: Vec<OperationId>,
    /// Values rewritten to point at an equivalent, as `(from, to)`.
    pub rewritten: Vec<(ValueId, ValueId)>,
    /// Anything the pass wants to say — in particular, transformations it
    /// declined and why.
    pub notes: Diagnostics,
}

impl PassReport {
    /// An empty report for a pass that has not changed anything yet.
    pub fn new(pass: &str) -> Self {
        PassReport {
            pass: pass.to_string(),
            ..Default::default()
        }
    }

    /// Records a removed operation.
    pub fn removed(&mut self, op: OperationId) {
        self.removed.push(op);
        self.changed = true;
    }

    /// Records a created operation.
    pub fn created(&mut self, op: OperationId) {
        self.created.push(op);
        self.changed = true;
    }

    /// Records a rewritten value.
    pub fn rewrote(&mut self, from: ValueId, to: ValueId) {
        self.rewritten.push((from, to));
        self.changed = true;
    }

    /// How many individual edits the pass made.
    pub fn edit_count(&self) -> usize {
        self.removed.len() + self.created.len() + self.rewritten.len()
    }
}

impl fmt::Display for PassReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.changed {
            return write!(f, "{}: no change", self.pass);
        }
        write!(f, "{}:", self.pass)?;
        if !self.removed.is_empty() {
            write!(f, " removed {}", self.removed.len())?;
        }
        if !self.created.is_empty() {
            write!(f, " created {}", self.created.len())?;
        }
        if !self.rewritten.is_empty() {
            write!(f, " rewrote {}", self.rewritten.len())?;
        }
        Ok(())
    }
}

/// The outcome of running a whole pipeline.
#[derive(Clone, Debug, Default)]
pub struct PipelineReport {
    /// How many fixpoint rounds ran.
    pub rounds: usize,
    /// Every pass report, in the order the passes ran.
    pub reports: Vec<PassReport>,
    /// Whether any pass changed anything.
    pub changed: bool,
    /// Set when the pipeline hit its iteration cap instead of converging.
    pub hit_iteration_cap: bool,
}

impl PipelineReport {
    /// The total number of edits across every pass.
    pub fn edit_count(&self) -> usize {
        self.reports.iter().map(PassReport::edit_count).sum()
    }

    /// The reports produced by one named pass.
    pub fn by_pass<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a PassReport> {
        self.reports.iter().filter(move |r| r.pass == name)
    }
}

impl fmt::Display for PipelineReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} round(s), {} edit(s){}",
            self.rounds,
            self.edit_count(),
            if self.hit_iteration_cap {
                ", stopped at the iteration cap"
            } else {
                ""
            }
        )?;
        for report in self.reports.iter().filter(|r| r.changed) {
            writeln!(f, "  {report}")?;
        }
        Ok(())
    }
}

/// Runs a sequence of passes to a fixpoint.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    max_rounds: usize,
}

impl PassManager {
    /// An empty pipeline.
    pub fn new() -> Self {
        PassManager {
            passes: Vec::new(),
            max_rounds: 8,
        }
    }

    /// The default v0.1 pipeline.
    ///
    /// Deduplication runs first because it turns duplicate calls into dead
    /// ones; elimination then removes them; scheduling comes last, once the
    /// block holds only the operations that survive.
    pub fn default_pipeline() -> Self {
        PassManager::new()
            .with(crate::ToolCallDeduplication)
            .with(crate::DeadActionElimination)
            .with(crate::Parallelization)
    }

    /// Appends a pass.
    pub fn with(mut self, pass: impl Pass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Sets how many fixpoint rounds are allowed before the manager gives up.
    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds.max(1);
        self
    }

    /// The passes in the pipeline, in order.
    pub fn passes(&self) -> impl Iterator<Item = &dyn Pass> {
        self.passes.iter().map(AsRef::as_ref)
    }

    /// Runs every pass repeatedly until a full round changes nothing.
    pub fn run(&self, module: &mut Module) -> PipelineReport {
        let mut pipeline = PipelineReport::default();
        for round in 1..=self.max_rounds {
            pipeline.rounds = round;
            let mut changed_this_round = false;
            for pass in &self.passes {
                let report = pass.run(module);
                changed_this_round |= report.changed;
                pipeline.changed |= report.changed;
                pipeline.reports.push(report);
            }
            if !changed_this_round {
                return pipeline;
            }
        }
        pipeline.hit_iteration_cap = true;
        pipeline
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::default_pipeline()
    }
}

impl fmt::Debug for PassManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PassManager")
            .field(
                "passes",
                &self.passes.iter().map(|p| p.name()).collect::<Vec<_>>(),
            )
            .field("max_rounds", &self.max_rounds)
            .finish()
    }
}
