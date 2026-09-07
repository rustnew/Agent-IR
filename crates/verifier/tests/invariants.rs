//! The negative test suite of §12, phase 3: for every invariant, a program
//! that breaks it and the exact diagnostic that must come back.
//!
//! The positive side matters just as much — a verifier that rejects everything
//! passes every negative test — so the examples in `examples/` are checked to
//! verify cleanly in both phases.

use agent_ir_core::{Diagnostics, Module};
use agent_ir_parser::parse_module;
use agent_ir_verifier::Verifier;

fn module(source: &str) -> Module {
    parse_module(source).unwrap_or_else(|e| panic!("test fixture should parse: {e}"))
}

fn check(source: &str) -> Diagnostics {
    Verifier::new().verify(&module(source))
}

fn codes(report: &Diagnostics) -> Vec<String> {
    report.errors().map(|d| d.code.clone()).collect()
}

fn assert_rejects(source: &str, code: &str) -> Diagnostics {
    let report = check(source);
    assert!(
        report.errors().any(|d| d.code == code),
        "expected a {code} error, got {:?}\n{report}",
        codes(&report)
    );
    report
}

fn assert_accepts(source: &str) {
    let report = check(source);
    assert!(!report.has_errors(), "expected no errors, got:\n{report}");
}

// ---------------------------------------------------------------------- I1

#[test]
fn i1_rejects_an_operand_with_no_dominating_producer() {
    // The parser enforces this too, so the module has to be built such that a
    // value defined inside a region is referenced from outside it. Passes can
    // produce exactly this shape by moving an operation, which is why the
    // verifier must catch it independently.
    let mut m = module(
        r#"module @m version(0) {
  agent.func "f" {
    %outer = core.constant {effect = #pure, value = 1} : !core.int
    %user = core.cast(%outer) {effect = #pure} : !core.int
    %par = control.parallel {
      %inner = core.constant {effect = #pure, value = 2} : !core.int
      control.yield(%inner) {effect = #pure}
    } {effect = #pure} : !core.int
    agent.return(%user) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    let inner = m
        .all_values()
        .find(|v| v.name.as_deref() == Some("inner"))
        .unwrap()
        .id;
    let user = m
        .op_ids()
        .into_iter()
        .find(|&id| m.op(id).name.is("core", "cast"))
        .unwrap();
    m.op_mut(user).operands = vec![inner];

    let report = Verifier::new().verify(&m);
    assert!(
        report.errors().any(|d| d.code == "I1"),
        "expected I1, got {:?}\n{report}",
        codes(&report)
    );
}

#[test]
fn i1_rejects_an_unresolved_type_crossing_an_effect_boundary() {
    assert_rejects(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
    %guess = agent.action "guess"() {effect = #pure} : !core.unknown
    %page = tool.call "fetch"(%guess) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#,
        "I1",
    );
}

#[test]
fn i1_allows_an_unresolved_type_between_pure_operations() {
    assert_accepts(
        r#"module @m version(0) {
  agent.func "f" {
    %guess = agent.action "guess"() {effect = #pure} : !core.unknown
    %typed = core.cast(%guess) {effect = #pure} : !core.int
    agent.return(%typed) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
}

// ---------------------------------------------------------------------- I2

#[test]
fn i2_rejects_an_unguarded_irreversible_action() {
    let report = assert_rejects(
        r#"module @m version(0) {
  capability @wipe scope("db") grants(irreversible)

  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I2",
    );
    let diagnostic = report.errors().find(|d| d.code == "I2").unwrap();
    assert!(diagnostic.suggestion.is_some(), "an I2 rejection must say what to do");
    assert!(diagnostic.operation.is_some(), "an I2 rejection must name the operation");
}

#[test]
fn i2_accepts_an_irreversible_action_behind_a_matching_verify() {
    assert_accepts(
        r#"module @m version(0) {
  capability @wipe scope("db") grants(irreversible)

  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    agent.verify(%db) {effect = #pure, capability = "wipe"}
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    );
}

#[test]
fn i2_rejects_a_verify_that_names_the_wrong_scope() {
    assert_rejects(
        r#"module @m version(0) {
  capability @wipe_cache scope("cache") grants(irreversible)

  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    agent.verify(%db) {effect = #pure, capability = "wipe_cache"}
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I2",
    );
}

#[test]
fn i2_rejects_a_verify_that_comes_after_the_action() {
    assert_rejects(
        r#"module @m version(0) {
  capability @wipe scope("db") grants(irreversible)

  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.verify(%db) {effect = #pure, capability = "wipe"}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I2",
    );
}

#[test]
fn i2_rejects_a_verify_for_a_capability_awaiting_approval() {
    assert_rejects(
        r#"module @m version(0) {
  capability @wipe scope("db") grants(irreversible) requires_approval

  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    agent.verify(%db) {effect = #pure, capability = "wipe"}
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I2",
    );
}

// ---------------------------------------------------------------------- I3

#[test]
fn i3_rejects_two_writes_to_the_same_scope_in_one_parallel_region() {
    assert_rejects(
        r#"module @m version(0) {
  capability @db scope("db") grants(write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    control.parallel {
      tool.call "insert"(%row) {effect = #write_external<db>}
      tool.call "update"(%row) {effect = #write_external<db>}
    } {effect = #pure}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I3",
    );
}

#[test]
fn i3_rejects_a_write_racing_a_read_of_the_same_scope() {
    assert_rejects(
        r#"module @m version(0) {
  capability @db scope("db") grants(write_external, read_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    control.parallel {
      tool.call "insert"(%row) {effect = #write_external<db>}
      %seen = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    } {effect = #pure}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I3",
    );
}

#[test]
fn i3_accepts_concurrent_writes_to_different_scopes() {
    assert_accepts(
        r#"module @m version(0) {
  capability @db scope("db") grants(write_external)
  capability @cache scope("cache") grants(write_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    control.parallel {
      tool.call "insert"(%row) {effect = #write_external<db>}
      tool.call "evict"(%row) {effect = #write_external<cache>}
    } {effect = #pure}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    );
}

#[test]
fn i3_sees_a_conflict_hidden_inside_a_nested_region() {
    assert_rejects(
        r#"module @m version(0) {
  capability @db scope("db") grants(write_external, read_external)

  agent.func "f" {
  ^bb0(%row: !tool.ref<db>):
    control.parallel {
      control.loop(%row) {
      ^bb0(%item: !tool.ref<db>):
        tool.call "insert"(%item) {effect = #write_external<db>}
      } {effect = #pure, max_iterations = 8}
      %seen = tool.call "select"(%row) {effect = #read_external<db>} : !tool.result<row>
    } {effect = #pure}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I3",
    );
}

// ---------------------------------------------------------------------- I4

#[test]
fn i4_rejects_a_low_confidence_llm_value_feeding_an_effectful_operation() {
    assert_rejects(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external, stochastic)

  agent.func "f" {
    %url = agent.action "invent_url"() {effect = #stochastic} : !core.string provenance(%url = llm confidence(0.4))
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#,
        "I4",
    );
}

#[test]
fn i4_accepts_the_same_value_once_it_has_been_verified() {
    assert_accepts(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external, stochastic)

  agent.func "f" {
    %url = agent.action "invent_url"() {effect = #stochastic} : !core.string provenance(%url = llm confidence(0.4))
    agent.verify(%url) {effect = #pure, capability = "web"}
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
}

#[test]
fn i4_accepts_a_high_confidence_llm_value() {
    assert_accepts(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external, stochastic)

  agent.func "f" {
    %url = agent.action "invent_url"() {effect = #stochastic} : !core.string provenance(%url = llm confidence(0.95))
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
}

#[test]
fn i4_looks_through_a_control_operation_that_declares_itself_pure() {
    // `control.loop` is `#pure` on its own line; its body calls a tool. The
    // guess still reaches the world, so I4 still applies.
    assert_rejects(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external, stochastic)

  agent.func "f" {
    %urls = agent.action "invent_urls"() {effect = #stochastic} : !tool.ref<urls> provenance(%urls = llm confidence(0.3))
    control.loop(%urls) {
    ^bb0(%url: !tool.ref<url>):
      %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    } {effect = #pure, max_iterations = 10}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I4",
    );
}

#[test]
fn the_confidence_threshold_is_configurable() {
    let source = r#"module @m version(0) {
  capability @web scope("web") grants(read_external, stochastic)

  agent.func "f" {
    %url = agent.action "invent_url"() {effect = #stochastic} : !core.string provenance(%url = llm confidence(0.5))
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#;
    let m = module(source);
    assert!(Verifier::new().verify(&m).has_errors());
    assert!(!Verifier::new()
        .with_confidence_threshold(0.2)
        .verify(&m)
        .has_errors());
}

// ---------------------------------------------------------------------- I5

#[test]
fn i5_rejects_a_loop_with_no_termination_guard() {
    // `max_iterations` is required by the dialect signature too, so this
    // reports both S3 and I5 — the point is that the loop guard is not
    // optional at either level.
    let report = check(
        r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %x = core.cast(%item) {effect = #pure} : !core.int
    } {effect = #pure}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    assert!(report.errors().any(|d| d.code == "I5"), "{report}");
    assert!(report.errors().any(|d| d.code == "S3"), "{report}");
}

#[test]
fn i5_rejects_a_non_positive_iteration_bound() {
    assert_rejects(
        r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%items: !tool.ref<items>):
    control.loop(%items) {
    ^bb0(%item: !tool.ref<item>):
      %x = core.cast(%item) {effect = #pure} : !core.int
    } {effect = #pure, max_iterations = 0}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "I5",
    );
}

// ------------------------------------------------------------ effect legality

#[test]
fn t2_rejects_an_operation_relabelling_its_effect() {
    // `core.constant` may only be `#pure`. Letting a program declare otherwise
    // would let it smuggle an effect past every §5 pass condition.
    assert_rejects(
        r#"module @m version(0) {
  capability @db scope("db") grants(irreversible)

  agent.func "f" {
    %x = core.constant {effect = #irreversible<db>, value = 1} : !core.int
    agent.return(%x) {effect = #pure}
  } {effect = #pure}
}
"#,
        "T2",
    );
}

#[test]
fn t2_rejects_a_tool_call_claiming_to_be_stochastic() {
    assert_rejects(
        r#"module @m version(0) {
  capability @llm scope(*) grants(stochastic)

  agent.func "f" {
    %x = tool.call "sample"() {effect = #stochastic} : !tool.result<sample>
    agent.return(%x) {effect = #pure}
  } {effect = #pure}
}
"#,
        "T2",
    );
}

// ------------------------------------------------------------------ structure

#[test]
fn an_unknown_operation_is_rejected() {
    assert_rejects(
        r#"module @m version(0) {
  agent.func "f" {
    quantum.entangle {effect = #pure}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
        "S1",
    );
}

#[test]
fn a_wrong_arity_is_rejected() {
    assert_rejects(
        r#"module @m version(0) {
  agent.func "f" {
    %a = core.constant {effect = #pure, value = 1} : !core.int
    %b = core.cast(%a, %a) {effect = #pure} : !core.int
    agent.return(%b) {effect = #pure}
  } {effect = #pure}
}
"#,
        "S2",
    );
}

#[test]
fn a_missing_required_attribute_is_rejected() {
    assert_rejects(
        r#"module @m version(0) {
  agent.func "f" {
    %a = core.constant {effect = #pure} : !core.int
    agent.return(%a) {effect = #pure}
  } {effect = #pure}
}
"#,
        "S3",
    );
}

#[test]
fn an_operation_after_a_terminator_is_rejected() {
    assert_rejects(
        r#"module @m version(0) {
  agent.func "f" {
    agent.return {effect = #pure}
    %a = core.constant {effect = #pure, value = 1} : !core.int
  } {effect = #pure}
}
"#,
        "S5",
    );
}

#[test]
fn a_region_producing_results_must_yield_them() {
    assert_rejects(
        r#"module @m version(0) {
  agent.func "f" {
    %x = control.parallel {
      %inner = core.constant {effect = #pure, value = 1} : !core.int
    } {effect = #pure} : !core.int
    agent.return(%x) {effect = #pure}
  } {effect = #pure}
}
"#,
        "S5",
    );
}

#[test]
fn only_functions_may_sit_at_module_level() {
    assert_rejects(
        r#"module @m version(0) {
  %stray = core.constant {effect = #pure, value = 1} : !core.int
}
"#,
        "S1",
    );
}

// -------------------------------------------------------------- safety phase

#[test]
fn c1_rejects_an_effect_the_agent_holds_no_capability_for() {
    let m = module(
        r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    %row = tool.call "select"(%db) {effect = #read_external<db>} : !tool.result<row>
    agent.return(%row) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    // Well formed...
    assert!(!Verifier::new().verify(&m).has_errors());
    // ...but not authorized. The two rejection points are distinct (§4).
    let safety = Verifier::new().verify_safety(&m);
    assert!(safety.errors().any(|d| d.code == "C1"), "{safety}");
}

#[test]
fn c2_rejects_a_gated_capability_without_recorded_approval() {
    let mut m = module(
        r#"module @m version(0) {
  capability @wipe scope("db") grants(irreversible) requires_approval

  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    agent.verify(%db) {effect = #pure, capability = "wipe"}
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    let safety = Verifier::new().verify_safety(&m);
    assert!(safety.errors().any(|d| d.code == "C2"), "{safety}");

    m.capabilities.approve("wipe");
    assert!(!Verifier::new().verify_all(&m).has_errors());
}

#[test]
fn verify_all_stops_before_the_safety_phase_when_the_program_is_malformed() {
    let m = module(
        r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    let report = Verifier::new().verify_all(&m);
    assert!(report.errors().any(|d| d.code == "I2"));
    assert!(
        !report.errors().any(|d| d.code == "C1"),
        "the safety phase should not run on a malformed program:\n{report}"
    );
}

// ----------------------------------------------------------------- positives

#[test]
fn every_checked_in_example_verifies_in_both_phases() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut checked = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "air") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let m = parse_module(&source).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let report = Verifier::new().verify_all(&m);
        assert!(
            !report.has_errors(),
            "{} should verify cleanly:\n{report}",
            path.display()
        );
        checked += 1;
    }
    assert!(checked >= 2, "expected at least two examples, saw {checked}");
}

#[test]
fn a_clean_program_produces_no_diagnostics_at_all() {
    let report = check(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    assert!(report.is_empty(), "{report}");
}
