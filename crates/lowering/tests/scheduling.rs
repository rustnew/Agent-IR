//! Scheduler tests: batching, cost, budget degradation and the §13.1 claim.

use agent_ir_core::Module;
use agent_ir_lowering::{
    scheduler::Estimator, Budget, Cost, ExecutionPlan, GenericRuntime, Scheduler,
};
use agent_ir_parser::parse_module;
use agent_ir_passes::PassManager;

fn module(source: &str) -> Module {
    parse_module(source).expect("fixture should parse")
}

fn schedule(source: &str) -> ExecutionPlan {
    Scheduler::new(GenericRuntime)
        .schedule(&module(source), "f")
        .expect("should schedule")
}

const THREE_INDEPENDENT: &str = r#"module @m version(0) {
  capability @host scope("host") grants(read_external)

  agent.func "f" {
  ^bb0(%a: !tool.ref<x>, %b: !tool.ref<y>, %c: !tool.ref<z>):
    %ia = tool.call "inspect_model"(%a) {effect = #read_external<host>} : !core.int
    %ib = tool.call "inspect_hardware"(%b) {effect = #read_external<host>} : !core.int
    %ic = tool.call "inspect_dataset"(%c) {effect = #read_external<host>} : !core.int
    %p = tool.call "profile"(%ia, %ib, %ic) {effect = #read_external<host>} : !core.int
    agent.return(%p) {effect = #pure}
  } {effect = #pure}
}
"#;

#[test]
fn independent_reads_land_in_one_batch() {
    let plan = schedule(THREE_INDEPENDENT);
    assert_eq!(plan.plan.batches[0].len(), 3, "{:#?}", plan.plan.batches[0]);
    assert_eq!(plan.plan.max_width(), 3);
}

#[test]
fn a_dependent_step_lands_in_a_later_batch() {
    let plan = schedule(THREE_INDEPENDENT);
    let profile_batch = plan
        .plan
        .batches
        .iter()
        .position(|batch| {
            batch
                .iter()
                .any(|s| s.literal.as_deref() == Some("profile"))
        })
        .unwrap();
    assert_eq!(profile_batch, 1);
}

#[test]
fn the_terminator_always_runs_alone_and_last() {
    let plan = schedule(THREE_INDEPENDENT);
    let last = plan.plan.batches.last().unwrap();
    assert_eq!(last.len(), 1);
    assert_eq!(last[0].name, "agent.return");
}

#[test]
fn batching_is_what_makes_the_section_13_1_latency_claim() {
    // Four reads at 250 ms each. Sequential: 1000 ms. Batched: three at once
    // then one, so 500 ms. This is the gain §13.1 predicts, stated as a number
    // the benchmark of §9.4 can later check against reality.
    let plan = schedule(THREE_INDEPENDENT);
    assert_eq!(plan.estimated.tool_calls, 4, "no work disappeared");
    assert_eq!(plan.estimated.latency_ms, 250 + 250 + 1);
}

#[test]
fn the_scheduler_finds_parallelism_the_pass_left_behind() {
    // `agent.verify` guards what follows it, so Parallelization refuses to move
    // it — which leaves the two independent reads on either side unfused. The
    // scheduler still batches all three, because it orders execution rather
    // than rewriting the program, and the guard's ordering is preserved by the
    // dependency graph rather than by textual position.
    let source = r#"module @m version(0) {
  capability @host scope("host") grants(read_external)

  agent.func "f" {
  ^bb0(%x: !tool.ref<x>):
    %a = tool.call "a"(%x) {effect = #read_external<host>} : !core.int
    agent.verify(%x) {effect = #pure, capability = "host"}
    %c = tool.call "c"(%x) {effect = #read_external<host>} : !core.int
    agent.return(%a, %c) {effect = #pure}
  } {effect = #pure}
}
"#;
    let mut m = module(source);
    let report = PassManager::default_pipeline().run(&mut m);
    assert!(
        !report.by_pass("parallelization").any(|r| r.changed),
        "the pass should decline to move a guard"
    );

    let plan = Scheduler::new(GenericRuntime).schedule(&m, "f").unwrap();
    let first = &plan.plan.batches[0];
    let names: Vec<&str> = first.iter().filter_map(|s| s.literal.as_deref()).collect();
    assert!(names.contains(&"a") && names.contains(&"c"), "{names:?}");
}

#[test]
fn a_conflicting_write_is_never_batched_with_the_read_it_races() {
    let plan = schedule(
        r#"module @m version(0) {
  capability @db scope("db") grants(read_external, write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    tool.call "update"(%row) {effect = #write_external<db>}
    %seen = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    agent.return(%seen) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    for batch in &plan.plan.batches {
        assert!(batch.len() <= 1, "I3 forbids batching these:\n{batch:#?}");
    }
}

#[test]
fn a_non_replayable_step_carries_an_idempotency_key() {
    let plan = schedule(
        r#"module @m version(0) {
  capability @pay scope("ledger") grants(irreversible)
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%invoice: !tool.ref<invoice>):
    %page = tool.call "fetch"(%invoice) {effect = #read_external<web>} : !tool.result<page>
    agent.verify(%invoice) {effect = #pure, capability = "pay"}
    tool.call "pay"(%invoice) {effect = #irreversible<ledger>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    let steps = plan.plan.steps();
    let pay = steps
        .iter()
        .find(|s| s.literal.as_deref() == Some("pay"))
        .unwrap();
    let fetch = steps
        .iter()
        .find(|s| s.literal.as_deref() == Some("fetch"))
        .unwrap();
    assert!(pay.idempotency_key.is_some(), "§8.3 requires a key here");
    assert!(
        fetch.idempotency_key.is_none(),
        "a replayable read needs no key"
    );
}

#[test]
fn a_loop_costs_its_body_times_its_bound() {
    let plan = schedule(
        r#"module @m version(0) {
  capability @bench scope("bench") grants(read_external)

  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %r = tool.call "benchmark"(%item) {effect = #read_external<bench>} : !tool.result<r>
    } {effect = #pure, max_iterations = 40}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    assert_eq!(plan.estimated.tool_calls, 40);
}

#[test]
fn an_over_budget_plan_degrades_by_halving_loop_bounds() {
    let source = r#"module @m version(0) {
  capability @bench scope("bench") grants(read_external)

  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    agent.budget {effect = #pure, tool_call_budget = 12}
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %r = tool.call "benchmark"(%item) {effect = #read_external<bench>} : !tool.result<r>
    } {effect = #pure, max_iterations = 40}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#;
    let plan = Scheduler::new(GenericRuntime)
        .schedule(&module(source), "f")
        .unwrap();
    assert!(plan.estimated.tool_calls <= 12, "{}", plan.estimated);
    assert!(
        plan.notes.iter().any(|d| d.code == "degraded"),
        "{:?}",
        plan.notes
    );
    assert!(
        !plan.notes.iter().any(|d| d.code == "budget"),
        "halving was enough; no human review needed"
    );
}

#[test]
fn a_budget_that_cannot_be_met_asks_for_human_review() {
    let source = r#"module @m version(0) {
  capability @bench scope("bench") grants(read_external)

  agent.func "f" {
  ^bb0(%item: !tool.ref<item>):
    agent.budget {effect = #pure, tool_call_budget = 0}
    %r = tool.call "benchmark"(%item) {effect = #read_external<bench>} : !tool.result<r>
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#;
    let plan = Scheduler::new(GenericRuntime)
        .schedule(&module(source), "f")
        .unwrap();
    let budget_note = plan.notes.iter().find(|d| d.code == "budget").unwrap();
    assert!(
        budget_note.message.contains("cannot be met"),
        "{budget_note}"
    );
    assert!(budget_note
        .suggestion
        .as_ref()
        .unwrap()
        .contains("HUMAN_REVIEW"));
}

#[test]
fn an_explicit_budget_argument_overrides_the_default() {
    let scheduler = Scheduler::new(GenericRuntime).with_budget(Budget {
        latency_budget_ms: Some(10),
        ..Budget::unlimited()
    });
    let plan = scheduler.schedule(&module(THREE_INDEPENDENT), "f").unwrap();
    assert!(plan.notes.iter().any(|d| d.code == "budget"));
}

#[test]
fn per_operation_estimates_override_the_defaults() {
    let plan = schedule(
        r#"module @m version(0) {
  capability @slow scope("slow") grants(read_external)

  agent.func "f" {
  ^bb0(%x: !tool.ref<x>):
    %r = tool.call "slow_thing"(%x) {effect = #read_external<slow>, est_latency_ms = 9000} : !core.int
    agent.return(%r) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    assert_eq!(plan.estimated.latency_ms, 9000 + 1);
}

#[test]
fn a_custom_estimator_changes_the_numbers_but_not_the_shape() {
    let fast = Estimator {
        tool_call: Cost {
            tool_calls: 1,
            latency_ms: 1,
            ..Cost::ZERO
        },
        ..Estimator::default()
    };
    let plan = Scheduler::new(GenericRuntime)
        .with_estimator(fast)
        .schedule(&module(THREE_INDEPENDENT), "f")
        .unwrap();
    assert_eq!(plan.estimated.tool_calls, 4);
    assert_eq!(plan.estimated.latency_ms, 1 + 1 + 1);
}

#[test]
fn scheduling_a_missing_function_reports_rather_than_panics() {
    let errors = Scheduler::new(GenericRuntime)
        .schedule(&module(THREE_INDEPENDENT), "nonexistent")
        .unwrap_err();
    assert!(errors.errors().any(|d| d.code == "schedule"));
}

#[test]
fn the_plan_records_where_every_step_came_from() {
    let plan = schedule(THREE_INDEPENDENT);
    let m = module(THREE_INDEPENDENT);
    for step in plan.plan.steps() {
        assert_eq!(
            m.op(step.op).name.to_string(),
            step.name,
            "a step must point back at its operation"
        );
    }
}

#[test]
fn the_specification_example_schedules_end_to_end() {
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/optimize_inference.air"),
    )
    .unwrap();
    let mut m = parse_module(&source).unwrap();
    PassManager::default_pipeline().run(&mut m);

    let plan = Scheduler::new(GenericRuntime)
        .schedule(&m, "optimize_inference")
        .expect("the specification example should schedule");
    assert!(
        plan.estimated.llm_calls >= 1,
        "generate_candidates is a model call"
    );
    assert!(
        plan.estimated.tool_calls >= 40,
        "40 benchmarks: {}",
        plan.estimated
    );
}
