//! Tool Call Deduplication / Result Reuse (§5).
//!
//! **Validity condition.** A later operation may reuse an earlier one's results
//! when
//!
//! 1. both have the same cache key — name, literal, operands, effect scope and
//!    every other attribute,
//! 2. both effect summaries are replayable (`Pure` or `ReadExternal`), and
//! 3. nothing between them writes a scope either of them touches, and
//! 4. neither declares `no_cache = true`.
//!
//! **Risk if violated.** Reusing a stale result: a price, a system state, a
//! row that something wrote in between. Condition 3 is the static stand-in for
//! §5's "validity window not expired" — a compiler has no clock, so instead of
//! guessing a freshness bound it proves that nothing observable changed. An
//! operation whose freshness depends on wall-clock time rather than on writes
//! this program makes must opt out with `no_cache = true`.
//!
//! The pass only considers operations in the same block, so a read inside a
//! loop body is never assumed to hold across iterations.

use crate::manager::{Pass, PassReport};
use agent_ir_analysis::EffectSummary;
use agent_ir_core::{BlockId, CacheKey, Diagnostic, Module, OperationId};
use std::collections::HashMap;

/// Replaces a repeated read with the result of the first one.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolCallDeduplication;

impl Pass for ToolCallDeduplication {
    fn name(&self) -> &'static str {
        "tool-call-deduplication"
    }

    fn description(&self) -> &'static str {
        "Reuses an earlier replayable result when nothing in between could have invalidated it."
    }

    fn run(&self, module: &mut Module) -> PassReport {
        let mut report = PassReport::new(self.name());
        let effects = EffectSummary::build(module);

        for block in collect_blocks(module) {
            deduplicate_block(module, block, &effects, &mut report);
        }

        report
    }
}

fn collect_blocks(module: &Module) -> Vec<BlockId> {
    let mut blocks = Vec::new();
    for index in 0..module.region_count() {
        blocks.extend(module.region_by_index(index).blocks.iter().copied());
    }
    blocks
}

fn deduplicate_block(
    module: &mut Module,
    block: BlockId,
    effects: &EffectSummary,
    report: &mut PassReport,
) {
    let ops: Vec<OperationId> = module
        .block(block)
        .ops
        .iter()
        .copied()
        .filter(|&op| !module.op(op).erased)
        .collect();

    let mut seen: HashMap<CacheKey, OperationId> = HashMap::new();

    for (position, &op) in ops.iter().enumerate() {
        if !is_cacheable(module, op, effects) {
            // An operation that is not itself cacheable may still invalidate
            // earlier ones, which the interference check below handles; it just
            // never becomes a cache entry.
            continue;
        }

        let key = module.op(op).cache_key();
        match seen.get(&key).copied() {
            None => {
                seen.insert(key, op);
            }
            Some(earlier) => {
                let earlier_position = ops.iter().position(|&o| o == earlier).unwrap_or(0);
                if let Some(interferer) =
                    interference(module, &ops[earlier_position + 1..position], op, effects)
                {
                    report.notes.push(
                        Diagnostic::note(
                            "reuse-declined",
                            format!(
                                "`{}` repeats an earlier call, but `{}` may have changed the \
                                 result in between",
                                module.op(op).name,
                                module.op(interferer).name
                            ),
                        )
                        .at(op),
                    );
                    // The later call becomes the fresh entry: anything after it
                    // may reuse *it* rather than the stale one.
                    seen.insert(key, op);
                    continue;
                }

                let earlier_results = module.op(earlier).results.clone();
                let later_results = module.op(op).results.clone();
                if earlier_results.len() != later_results.len() {
                    continue;
                }
                for (&from, &to) in later_results.iter().zip(earlier_results.iter()) {
                    if module.value(from).ty != module.value(to).ty {
                        continue;
                    }
                    module.replace_all_uses(from, to);
                    report.rewrote(from, to);
                }
                report.notes.push(
                    Diagnostic::note(
                        "result-reuse",
                        format!(
                            "`{}`{} reuses the result of the identical earlier call",
                            module.op(op).name,
                            module
                                .op(op)
                                .literal
                                .as_deref()
                                .map(|l| format!(" \"{l}\""))
                                .unwrap_or_default()
                        ),
                    )
                    .at(op),
                );
                // The duplicate is now unread. Dead Action Elimination removes
                // it on the next round of the pipeline, which keeps the two
                // §5 rows measurable apart.
            }
        }
    }
}

fn is_cacheable(module: &Module, op: OperationId, effects: &EffectSummary) -> bool {
    let operation = module.op(op);
    if operation.results.is_empty() {
        return false;
    }
    if operation.bool_attr("no_cache") == Some(true) {
        return false;
    }
    // A region-owning operation is never a cache entry: two loops with the same
    // operands are not the same computation.
    if !operation.regions.is_empty() {
        return false;
    }
    effects.is_replayable(op)
}

/// The first operation in `between` that could have changed what `op` reads.
fn interference(
    module: &Module,
    between: &[OperationId],
    op: OperationId,
    effects: &EffectSummary,
) -> Option<OperationId> {
    between
        .iter()
        .copied()
        .find(|&candidate| !module.op(candidate).erased && effects.conflict(candidate, op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_parser::parse_module;

    fn run(source: &str) -> (Module, PassReport) {
        let mut module = parse_module(source).expect("fixture should parse");
        let report = ToolCallDeduplication.run(&mut module);
        (module, report)
    }

    #[test]
    fn reuses_an_identical_external_read() {
        let (module, report) = run(r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %first = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    %second = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%second) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert_eq!(report.rewritten.len(), 1);
        assert!(
            module.to_string().contains("agent.return(%first)"),
            "the return should read the first call:\n{module}"
        );
    }

    #[test]
    fn refuses_when_a_write_to_the_same_scope_sits_in_between() {
        let (_, report) = run(r#"module @m version(0) {
  capability @db scope("db") grants(read_external, write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    %first = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    tool.call "update"(%row) {effect = #write_external<db>}
    %second = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    agent.return(%second) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(!report.changed);
        assert!(report.notes.iter().any(|d| d.code == "reuse-declined"));
    }

    #[test]
    fn allows_reuse_across_a_write_to_a_different_scope() {
        let (_, report) = run(r#"module @m version(0) {
  capability @db scope("db") grants(read_external)
  capability @cache scope("cache") grants(write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    %first = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    tool.call "evict"(%row) {effect = #write_external<cache>}
    %second = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    agent.return(%second) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert_eq!(report.rewritten.len(), 1);
    }

    #[test]
    fn never_deduplicates_a_write() {
        let (_, report) = run(r#"module @m version(0) {
  capability @db scope("db") grants(write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    %a = tool.call "append"(%row) {effect = #write_external<db>} : !tool.result<receipt>
    %b = tool.call "append"(%row) {effect = #write_external<db>} : !tool.result<receipt>
    agent.return(%b) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(!report.changed, "two appends are two appends");
    }

    #[test]
    fn never_deduplicates_a_stochastic_call() {
        let (_, report) = run(r#"module @m version(0) {
  capability @llm scope(*) grants(stochastic)

  agent.func "f" {
  ^bb0(%prompt: !core.string):
    %a = agent.action "sample"(%prompt) {effect = #stochastic} : !core.string
    %b = agent.action "sample"(%prompt) {effect = #stochastic} : !core.string
    agent.return(%b) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(
            !report.changed,
            "sampling twice is the point of sampling twice"
        );
    }

    #[test]
    fn different_arguments_are_different_calls() {
        let (_, report) = run(r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%a: !core.string, %b: !core.string):
    %first = tool.call "fetch"(%a) {effect = #read_external<web>} : !tool.result<page>
    %second = tool.call "fetch"(%b) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%second) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(!report.changed);
    }

    #[test]
    fn no_cache_opts_an_operation_out() {
        let (_, report) = run(r#"module @m version(0) {
  capability @clock scope("clock") grants(read_external)

  agent.func "f" {
    %a = tool.call "now"() {effect = #read_external<clock>, no_cache = true} : !core.int
    %b = tool.call "now"() {effect = #read_external<clock>, no_cache = true} : !core.int
    agent.return(%b) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(!report.changed, "a clock is never twice the same");
    }

    #[test]
    fn does_not_reach_across_block_boundaries() {
        // The read inside the loop body runs once per iteration and must not be
        // folded into the one before the loop.
        let (_, report) = run(r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string, %items: !tool.ref<items>):
    %outer = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %inner = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    } {effect = #pure, max_iterations = 10}
    agent.return(%outer) {effect = #pure}
  } {effect = #pure}
}
"#);
        assert!(!report.changed);
    }
}
