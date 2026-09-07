//! Pipeline-level tests: the passes composing, reaching a fixpoint, and never
//! producing a module the verifier rejects.

use agent_ir_core::Module;
use agent_ir_parser::parse_module;
use agent_ir_passes::{
    DeadActionElimination, Parallelization, Pass, PassManager, ToolCallDeduplication,
};
use agent_ir_verifier::Verifier;

fn optimized(source: &str) -> (Module, agent_ir_passes::PipelineReport) {
    let mut module = parse_module(source).expect("fixture should parse");
    let before = Verifier::new().verify_all(&module);
    assert!(!before.has_errors(), "fixture should verify first:\n{before}");

    let report = PassManager::default_pipeline().run(&mut module);

    let after = Verifier::new().verify_all(&module);
    assert!(
        !after.has_errors(),
        "the pipeline produced an invalid module:\n{module}\n{after}"
    );
    (module, report)
}

#[test]
fn deduplication_feeds_elimination() {
    // The second `fetch` is folded into the first, which leaves it unread,
    // which lets Dead Action Elimination remove it — a result no single pass
    // reaches alone.
    let (module, report) = optimized(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %first = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    %second = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    agent.return(%second) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    assert_eq!(
        module.to_string().matches("tool.call").count(),
        1,
        "one call should survive:\n{module}"
    );
    assert!(report.by_pass("tool-call-deduplication").any(|r| r.changed));
    assert!(report.by_pass("dead-action-elimination").any(|r| r.changed));
}

#[test]
fn the_pipeline_reaches_a_fixpoint_without_hitting_the_cap() {
    let (_, report) = optimized(
        r#"module @m version(0) {
  capability @web scope("web") grants(read_external)

  agent.func "f" {
  ^bb0(%url: !core.string):
    %a = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    %b = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    %c = tool.call "fetch"(%url) {effect = #read_external<web>} : !tool.result<page>
    %d = agent.action "render"(%a, %b, %c) {effect = #pure} : !core.string
    agent.return(%d) {effect = #pure}
  } {effect = #pure}
}
"#,
    );
    assert!(!report.hit_iteration_cap, "{report}");
}

#[test]
fn running_the_pipeline_twice_changes_nothing_the_second_time() {
    let source = std::fs::read_to_string(example("simple_agent.air")).unwrap();
    let mut module = parse_module(&source).unwrap();

    let first = PassManager::default_pipeline().run(&mut module);
    assert!(first.changed);
    let after_once = module.to_string();

    let second = PassManager::default_pipeline().run(&mut module);
    assert!(!second.changed, "the pipeline is not idempotent:\n{module}");
    assert_eq!(after_once, module.to_string());
}

#[test]
fn the_section_13_1_program_becomes_the_section_13_1_result() {
    let source = std::fs::read_to_string(example("simple_agent.air")).unwrap();
    let (module, _) = optimized(&source);
    let printed = module.to_string();

    // IR₁ of §13.1: the three inspections inside one parallel region, profile
    // after it.
    assert_eq!(printed.matches("control.parallel").count(), 1, "{printed}");
    let region = &printed[printed.find("control.parallel").unwrap()
        ..printed.find("control.yield").unwrap()];
    for literal in ["inspect_model", "inspect_hardware", "inspect_dataset"] {
        assert!(region.contains(literal), "{literal} should be parallel:\n{printed}");
    }
    assert!(!region.contains("profile"));
}

#[test]
fn the_specification_example_survives_the_pipeline() {
    let source = std::fs::read_to_string(example("optimize_inference.air")).unwrap();
    let (module, _) = optimized(&source);
    // Nothing effectful may vanish from a program that was already minimal in
    // that respect.
    let printed = module.to_string();
    for literal in ["benchmark", "generate_candidates", "select"] {
        assert!(printed.contains(literal), "{literal} disappeared:\n{printed}");
    }
}

#[test]
fn a_pipeline_of_one_pass_reports_only_that_pass() {
    let mut module = parse_module(
        r#"module @m version(0) {
  agent.func "f" {
    %dead = agent.action "compute"() {effect = #pure} : !core.int
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    )
    .unwrap();
    let report = PassManager::new().with(DeadActionElimination).run(&mut module);
    assert!(report.changed);
    assert!(report.reports.iter().all(|r| r.pass == "dead-action-elimination"));
}

#[test]
fn every_pass_names_and_describes_itself() {
    let passes: Vec<Box<dyn Pass>> = vec![
        Box::new(DeadActionElimination),
        Box::new(ToolCallDeduplication),
        Box::new(Parallelization),
    ];
    for pass in passes {
        assert!(!pass.name().is_empty());
        assert!(pass.description().ends_with('.'), "{}", pass.name());
        assert!(
            pass.name().chars().all(|c| c.is_ascii_lowercase() || c == '-'),
            "{} should be kebab-case",
            pass.name()
        );
    }
}

#[test]
fn an_empty_pipeline_changes_nothing() {
    let source = std::fs::read_to_string(example("simple_agent.air")).unwrap();
    let mut module = parse_module(&source).unwrap();
    let before = module.to_string();
    let report = PassManager::new().run(&mut module);
    assert!(!report.changed);
    assert_eq!(before, module.to_string());
}

fn example(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name)
}
