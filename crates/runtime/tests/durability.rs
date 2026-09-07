//! End-to-end runtime tests: execution, the semantic non-regression criterion
//! of §12 phase 5, the crash and recovery of §16, and the loop guards of §8.4.

use agent_ir_core::Module;
use agent_ir_lowering::{ExecutionPlan, GenericRuntime, Scheduler};
use agent_ir_parser::parse_module;
use agent_ir_passes::PassManager;
use agent_ir_runtime::{
    CheckpointStore, EnvError, EventLog, Executor, InMemoryCheckpointStore, InMemoryEventLog,
    Invocation, LoopGuardAction,
    RecordingEnvironment, RecoveryManager, RuntimeError, RuntimePolicy, Value,
};
use agent_ir_verifier::Verifier;

fn example(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples")
            .join(name),
    )
    .unwrap()
}

fn compile(source: &str, function: &str, optimize: bool) -> (Module, ExecutionPlan) {
    let mut module = parse_module(source).expect("should parse");
    let report = Verifier::new().verify_all(&module);
    assert!(!report.has_errors(), "should verify:\n{report}");
    if optimize {
        PassManager::default_pipeline().run(&mut module);
        let after = Verifier::new().verify_all(&module);
        assert!(!after.has_errors(), "optimizing broke it:\n{module}\n{after}");
    }
    let plan = Scheduler::new(GenericRuntime)
        .schedule(&module, function)
        .expect("should schedule");
    (module, plan)
}

fn candidates(count: usize) -> Value {
    Value::list((0..count).map(|i| Value::Str(format!("candidate-{i}"))))
}

fn benchmarking_environment() -> RecordingEnvironment {
    RecordingEnvironment::new()
        .on("benchmark", |call: &Invocation| {
            let name = call.arg(0).and_then(Value::as_str).unwrap_or("?").to_string();
            let latency = 100.0 + (name.len() as f64);
            Ok(Value::record([
                ("latency", Value::Float(latency)),
                ("accuracy", Value::Float(0.99)),
            ]))
        })
        .on("record_result", |_| Ok(Value::Null))
}

// ------------------------------------------------------------------ execution

#[test]
fn a_simple_program_runs_and_returns() {
    let (_, plan) = compile(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    %title = tool.result(%page) {effect = #pure, field = "title"} : !core.string
    agent.return(%title) {effect = #pure}
  } {effect = #pure}
}
"#,
        "f",
        false,
    );

    let mut env = RecordingEnvironment::new().returning(
        "fetch",
        Value::record([("title", Value::Str("Agent IR".into()))]),
    );
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();

    let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, vec![Value::Str("https://example.com".into())])
        .unwrap();

    assert_eq!(outcome.result, Value::Str("Agent IR".into()));
    assert_eq!(outcome.effects, 1);
    assert_eq!(env.call_names(), vec!["fetch"]);
}

#[test]
fn control_flow_drives_the_nested_plans() {
    let (_, plan) = compile(
        r#"module @m version(0) {
  capability @bench scope("bench") grants(read_external)

  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    %limit = core.constant {effect = #pure, value = 105.0} : !core.float
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %r = tool.call "benchmark"(%item) {effect = #read_external<bench>} : !tool.result<r>
      %latency = observation.metric "latency"(%r) {effect = #pure} : !core.float
      %too_slow = core.cmp "gt"(%latency, %limit) {effect = #pure} : !core.bool
      control.if(%too_slow) {
        agent.reject(%item) {effect = #pure}
      } {effect = #pure}
    } {effect = #pure, max_iterations = 8}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "f",
        false,
    );

    let mut env = benchmarking_environment();
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, vec![candidates(3)])
        .unwrap();

    assert_eq!(outcome.effects, 3, "one benchmark per candidate");
    // "candidate-0" is 11 characters, so latency 111 > 105: every candidate is
    // rejected, and the runtime recorded which ones.
    assert_eq!(outcome.state.rejected.len(), 3);
}

#[test]
fn a_loop_that_exceeds_its_declared_bound_is_an_error() {
    // Invariant I5 is a promise the program made. Silently truncating would
    // turn a broken promise into a wrong answer.
    let (_, plan) = compile(
        r#"module @m version(0) {
  capability @bench scope("bench") grants(read_external)

  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %r = tool.call "benchmark"(%item) {effect = #read_external<bench>} : !tool.result<r>
    } {effect = #pure, max_iterations = 2}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "f",
        false,
    );

    let mut env = benchmarking_environment();
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    let error = Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, vec![candidates(5)])
        .unwrap_err();
    assert!(matches!(error, RuntimeError::IterationLimit { limit: 2 }), "{error}");
}

#[test]
fn the_event_log_points_every_effect_back_at_its_operation() {
    let (module, plan) = compile(&example("durable_benchmark.air"), "benchmark_candidates", false);
    let mut env = benchmarking_environment();
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, vec![candidates(4)])
        .unwrap();

    let anchored = log.entries().iter().filter(|e| e.op.is_some()).count();
    assert!(anchored >= 8, "4 benchmarks + 4 records should be anchored");
    for event in log.entries().iter().filter_map(|e| e.op) {
        // Every anchor must name a real operation — that is what makes the log
        // an audit trail rather than a pile of strings.
        assert!(!module.op(event).name.dialect.is_empty());
    }
}

// -------------------------------------------- §12 phase 5: no semantic change

/// Runs a program twice — unoptimized and optimized — and returns both traces.
fn trace_both_ways(source: &str, function: &str, args: Vec<Value>) -> (Vec<String>, Vec<String>) {
    let mut traces = Vec::new();
    for optimize in [false, true] {
        let (_, plan) = compile(source, function, optimize);
        let mut env = benchmarking_environment()
            .returning("inspect_model", Value::Int(1))
            .returning("inspect_hardware", Value::Int(2))
            .returning("inspect_dataset", Value::Int(3))
            .returning("profile", Value::Int(6))
            .returning("fetch", Value::Str("page".into()));
        let mut log = InMemoryEventLog::new();
        let mut checkpoints = InMemoryCheckpointStore::new();
        let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
            .run(&plan, args.clone())
            .unwrap();
        let mut trace: Vec<String> = env
            .calls
            .iter()
            .map(|c| format!("{}({})", c.target, c.args.len()))
            .collect();
        trace.push(format!("=> {}", outcome.result));
        traces.push(trace);
    }
    let optimized = traces.pop().unwrap();
    (traces.pop().unwrap(), optimized)
}

#[test]
fn optimizing_does_not_change_what_the_program_does() {
    // The phase-5 success criterion of §12: "no pass alters the final result on
    // the test bench". The effects that reached the world and the value that
    // came back must be identical.
    let (plain, optimized) = trace_both_ways(
        &example("simple_agent.air"),
        "inspect_and_profile",
        vec![Value::Str("m".into()), Value::Str("h".into()), Value::Str("d".into())],
    );
    assert_eq!(plain, optimized, "optimizing changed the observable behaviour");
    assert_eq!(plain.last().unwrap(), "=> 6");
}

#[test]
fn deduplication_removes_a_call_without_changing_the_answer() {
    // Here the traces are deliberately *not* equal — that is the point of the
    // pass — but the answer must be.
    let source = r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %a = tool.call "fetch"(%url) {effect = #read_external<web>} : !core.string
    %b = tool.call "fetch"(%url) {effect = #read_external<web>} : !core.string
    agent.return(%b) {effect = #pure}
  } {effect = #pure}
}
"#;
    let (plain, optimized) = trace_both_ways(source, "f", vec![Value::Str("u".into())]);
    assert_eq!(plain.len(), 3, "two fetches and a result");
    assert_eq!(optimized.len(), 2, "one fetch and a result");
    assert_eq!(plain.last(), optimized.last(), "the answer must not move");
}

#[test]
fn optimizing_never_removes_an_effect_that_reached_the_world() {
    let source = r#"module @m version(0) {
  capability @db scope("db") grants(write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    %receipt = tool.call "record_result"(%row) {effect = #write_external<db>} : !core.string
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#;
    let (plain, optimized) = trace_both_ways(source, "f", vec![Value::Str("r".into())]);
    assert_eq!(plain, optimized);
    assert!(plain.iter().any(|line| line.starts_with("record_result")));
}

// ------------------------------------------------------- §16: crash and resume

#[test]
fn a_crash_mid_run_loses_nothing_and_repeats_no_write() {
    let (_, plan) = compile(&example("durable_benchmark.air"), "benchmark_candidates", true);
    let all = candidates(40);

    // --- first run: the environment dies after the 15th recorded result.
    let mut env = benchmarking_environment().crashing_after_writes(15);
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    let error = Executor::new(&mut env, &mut log, &mut checkpoints)
        .with_policy(RuntimePolicy { checkpoint_every: 5, ..RuntimePolicy::default() })
        .run(&plan, vec![all.clone()])
        .unwrap_err();
    assert!(matches!(error, RuntimeError::Environment(EnvError::Crashed(_))), "{error}");

    let writes_before = env.writing_calls().len();
    assert_eq!(writes_before, 15, "the fifteenth write happened, then the crash");
    assert!(checkpoints.latest().is_some(), "checkpoints should have been taken");

    // --- recovery: last checkpoint, plus the effects the log knows about.
    let resumption = RecoveryManager::new().resume(&log, &checkpoints);
    assert!(resumption.from_checkpoint.is_some());
    assert!(
        resumption.recovered_from_log > 0,
        "the writes after the last checkpoint must come back from the log"
    );
    RecoveryManager::new().note_resumption(&mut log, &resumption);

    // --- second run: the same plan, starting from the recovered ledger.
    //
    // The agent process died; the tool runtime did not. Reusing the same
    // environment is what makes this the §16 scenario rather than a fresh
    // world where nothing ever happened.
    env.crash_after_writes = None;
    let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
        .with_policy(RuntimePolicy { checkpoint_every: 5, ..RuntimePolicy::default() })
        .resuming_from(resumption.state)
        .run(&plan, vec![all])
        .unwrap();

    // Every candidate ends up recorded exactly once, across both runs.
    assert_eq!(
        env.writing_calls().len(),
        40,
        "40 candidates, 40 writes, no more and no less"
    );
    assert_eq!(
        outcome.replayed, 15,
        "the fifteen writes that already happened were skipped, not repeated"
    );
    // Fourteen came back through the executor's own ledger. The fifteenth —
    // the one the crash swallowed the acknowledgement for — could only come
    // from the tool runtime, which is why §8.3 puts the check there too.

    // The benchmark is a read. Replaying it is free, so recovery simply re-runs
    // it — which is exactly what `ReadExternal` licenses.
    let reruns = env.calls.iter().filter(|c| c.target == "benchmark").count();
    assert_eq!(reruns, 55, "15 before the crash, then all 40 again on replay");
}

#[test]
fn resuming_with_no_checkpoint_rebuilds_from_the_log_alone() {
    let (_, plan) = compile(&example("durable_benchmark.air"), "benchmark_candidates", false);
    let all = candidates(6);

    let mut env = benchmarking_environment().crashing_after_writes(3);
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    Executor::new(&mut env, &mut log, &mut checkpoints)
        .with_policy(RuntimePolicy { checkpoint_every: 0, ..RuntimePolicy::default() })
        .run(&plan, vec![all.clone()])
        .unwrap_err();
    assert!(checkpoints.latest().is_none(), "checkpoints were disabled");

    let resumption = RecoveryManager::new().resume(&log, &checkpoints);
    assert_eq!(resumption.from_checkpoint, None);
    assert_eq!(resumption.state.ledger.len(), 2, "two acknowledged writes");

    let mut env2 = benchmarking_environment();
    let outcome = Executor::new(&mut env2, &mut log, &mut checkpoints)
        .resuming_from(resumption.state)
        .run(&plan, vec![all])
        .unwrap();
    assert_eq!(outcome.replayed, 2);
    assert_eq!(env2.writing_calls().len(), 4);
}

#[test]
fn a_second_run_from_a_complete_ledger_touches_nothing() {
    let (_, plan) = compile(&example("durable_benchmark.air"), "benchmark_candidates", false);
    let all = candidates(5);

    let mut env = benchmarking_environment();
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    let first = Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, vec![all.clone()])
        .unwrap();
    assert_eq!(env.writing_calls().len(), 5);

    let mut env2 = benchmarking_environment();
    let second = Executor::new(&mut env2, &mut log, &mut checkpoints)
        .resuming_from(first.state)
        .run(&plan, vec![all])
        .unwrap();
    assert_eq!(env2.writing_calls().len(), 0, "nothing was written twice");
    assert_eq!(second.replayed, 5);
}

// ------------------------------------------------------------ §8.4 loop guards

#[test]
fn an_action_repeating_identically_trips_the_loop_guard() {
    let (_, plan) = compile(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %page = tool.call "fetch"(%item) {effect = #read_external<web>} : !core.string
    } {effect = #pure, max_iterations = 20}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "f",
        false,
    );

    // The same argument every time: the agent is stuck.
    let identical = Value::list((0..20).map(|_| Value::Str("same".into())));
    let mut env = RecordingEnvironment::new().returning("fetch", Value::Str("page".into()));
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();

    let error = Executor::new(&mut env, &mut log, &mut checkpoints)
        .with_policy(RuntimePolicy {
            repeat_threshold: 5,
            on_loop_guard: LoopGuardAction::ChangeStrategy,
            ..RuntimePolicy::default()
        })
        .run(&plan, vec![identical])
        .unwrap_err();

    match error {
        RuntimeError::LoopGuard { repeats, action, .. } => {
            assert_eq!(repeats, 6);
            assert_eq!(action, LoopGuardAction::ChangeStrategy);
        }
        other => panic!("expected a loop guard, got {other}"),
    }
    assert!(log
        .entries()
        .iter()
        .any(|e| matches!(e.kind, agent_ir_runtime::EventKind::LoopGuard { .. })));
}

#[test]
fn distinct_arguments_never_trip_the_guard() {
    let (_, plan) = compile(&example("durable_benchmark.air"), "benchmark_candidates", false);
    let mut env = benchmarking_environment();
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();

    let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
        .with_policy(RuntimePolicy { repeat_threshold: 3, ..RuntimePolicy::default() })
        .run(&plan, vec![candidates(30)])
        .unwrap();
    assert_eq!(outcome.effects, 60, "30 benchmarks and 30 records, all distinct");
}

// -------------------------------------------------------- the whole pipeline

#[test]
fn the_specification_example_runs_end_to_end() {
    let source = example("optimize_inference.air");
    let (_, plan) = compile(&source, "optimize_inference", true);

    let mut env = RecordingEnvironment::new()
        .returning("inspect_model", Value::Str("resnet".into()))
        .returning("inspect_hardware", Value::Str("a100".into()))
        .returning("profile", Value::Str("profile".into()))
        .returning(
            "generate_candidates",
            Value::list([Value::Str("c0".into()), Value::Str("c1".into())]),
        )
        .on("benchmark", |call: &Invocation| {
            let name = call.arg(0).and_then(Value::as_str).unwrap_or("?");
            Ok(Value::record([
                ("latency", Value::Float(10.0)),
                ("accuracy", Value::Float(if name == "c0" { 0.5 } else { 0.001 })),
            ]))
        })
        .returning("select", Value::Str("c1".into()));

    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, vec![Value::Str("model".into()), Value::Str("hw".into())])
        .unwrap();

    assert_eq!(outcome.result, Value::Str("c1".into()));
    // `c0` breaks the 0.01 accuracy constraint and is rejected; `c1` survives.
    assert_eq!(outcome.state.rejected.len(), 1);
    assert_eq!(
        outcome.state.rejected[0],
        Value::Str("c0".into()),
        "the wrong candidate was rejected"
    );
    // The two pure inspections are unused, so Dead Action Elimination removed
    // them before this ran; what is left must still be fully auditable.
    let effects: Vec<&str> = log
        .entries()
        .iter()
        .filter_map(|e| match &e.kind {
            agent_ir_runtime::EventKind::Effect { target, .. } => Some(target.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        effects,
        vec!["profile", "generate_candidates", "benchmark", "benchmark", "select"],
        "the audit trail should hold exactly the effects that happened"
    );
}

#[test]
fn an_environment_missing_a_tool_fails_loudly() {
    let (_, plan) = compile(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !core.string
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#,
        "f",
        false,
    );
    let mut env = RecordingEnvironment::new();
    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    let error = Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, vec![Value::Str("u".into())])
        .unwrap_err();
    assert!(matches!(error, RuntimeError::Environment(EnvError::Unknown(_))), "{error}");
    assert!(log
        .entries()
        .iter()
        .any(|e| matches!(e.kind, agent_ir_runtime::EventKind::Failed { .. })));
}
