//! Command line tests: every subcommand, on the checked-in examples, with the
//! exit codes a caller would branch on.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(relative)
}

fn agent_ir(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_agent-ir"))
        .args(args)
        .output()
        .expect("the binary should run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn example(name: &str) -> String {
    repo("examples").join(name).display().to_string()
}

// ----------------------------------------------------------------------- fmt

#[test]
fn every_checked_in_example_is_already_canonical() {
    let mut args = vec!["fmt".to_string(), "--check".to_string()];
    for entry in std::fs::read_dir(repo("examples")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "air") {
            args.push(path.display().to_string());
        }
    }
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = agent_ir(&refs);
    assert!(
        output.status.success(),
        "some examples are not canonical:\n{}",
        stderr(&output)
    );
}

#[test]
fn fmt_without_files_says_so() {
    let output = agent_ir(&["fmt"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("at least one file"));
}

// -------------------------------------------------------------------- verify

#[test]
fn verify_accepts_the_examples() {
    for name in ["optimize_inference.air", "simple_agent.air", "durable_benchmark.air"] {
        let output = agent_ir(&["verify", &example(name)]);
        assert!(output.status.success(), "{name}:\n{}", stderr(&output));
        assert!(stdout(&output).contains("accepted"));
    }
}

#[test]
fn verify_fails_with_a_diagnostic_and_a_nonzero_exit() {
    let path = std::env::temp_dir().join("agent-ir-cli-bad.air");
    std::fs::write(
        &path,
        r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%db: !tool.ref<db>):
    tool.call "delete_database"(%db) {effect = #irreversible<db>}
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#,
    )
    .unwrap();

    let output = agent_ir(&["verify", path.to_str().unwrap()]);
    assert!(!output.status.success(), "an unguarded deletion must not pass");
    assert!(stdout(&output).contains("I2"), "{}", stdout(&output));
    assert!(stdout(&output).contains("help:"), "a rejection must say what to do");
}

#[test]
fn verify_can_emit_json() {
    let output = agent_ir(&["verify", "--json", &example("simple_agent.air")]);
    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert!(parsed.is_object());
}

#[test]
fn the_static_phase_can_be_run_on_its_own() {
    // Well formed, but the agent holds no capability: static passes, the whole
    // thing does not. That is §4's two rejection points, visible from outside.
    let path = std::env::temp_dir().join("agent-ir-cli-unauthorized.air");
    std::fs::write(
        &path,
        r#"module @m version(0) {
  agent.func "f" {
  ^bb0(%url: !core.string):
    %page = tool.call "fetch"(%url) {effect = #read_external<web>} : !core.string
    agent.return(%page) {effect = #pure}
  } {effect = #pure}
}
"#,
    )
    .unwrap();

    let static_only = agent_ir(&["verify", "--static-only", path.to_str().unwrap()]);
    assert!(static_only.status.success(), "{}", stdout(&static_only));

    let both = agent_ir(&["verify", path.to_str().unwrap()]);
    assert!(!both.status.success());
    assert!(stdout(&both).contains("C1"), "{}", stdout(&both));
}

// ----------------------------------------------------------------------- opt

#[test]
fn opt_prints_a_program_that_still_verifies() {
    let output = agent_ir(&["opt", &example("simple_agent.air")]);
    assert!(output.status.success(), "{}", stderr(&output));
    let optimized = stdout(&output);
    assert!(optimized.contains("control.parallel"));

    let path = std::env::temp_dir().join("agent-ir-cli-optimized.air");
    std::fs::write(&path, &optimized).unwrap();
    let check = agent_ir(&["verify", path.to_str().unwrap()]);
    assert!(check.status.success(), "{}", stdout(&check));
}

#[test]
fn opt_report_explains_what_fired() {
    let output = agent_ir(&["opt", "--report", &example("simple_agent.air")]);
    assert!(output.status.success());
    let report = stdout(&output);
    assert!(report.contains("parallelization"), "{report}");
    assert!(report.contains("no dependency"), "{report}");
}

// ---------------------------------------------------------------------- plan

#[test]
fn plan_shows_the_batches_and_the_cost() {
    let output = agent_ir(&["plan", &example("simple_agent.air")]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("3 concurrent"), "{text}");
    assert!(text.contains("runtime.tool_call"), "{text}");
    assert!(text.contains("estimated:"), "{text}");
}

#[test]
fn plan_shows_idempotency_keys_where_they_are_required() {
    let output = agent_ir(&["plan", &example("durable_benchmark.air")]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("idempotency:"), "a write needs a key:\n{text}");
    assert!(text.contains("up to 40 iterations"), "{text}");
}

#[test]
fn plan_can_emit_json() {
    let output = agent_ir(&["plan", "--json", &example("simple_agent.air")]);
    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(parsed["function"], "inspect_and_profile");
}

#[test]
fn plan_asks_which_function_when_it_cannot_tell() {
    let output = agent_ir(&["plan", "--function", "nope", &example("simple_agent.air")]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("inspect_and_profile"), "{}", stderr(&output));
}

// ----------------------------------------------------------------------- run

#[test]
fn run_executes_the_program_against_stubbed_tools() {
    let output = agent_ir(&[
        "run",
        &example("simple_agent.air"),
        "--tools",
        &repo("examples/tools/inspect.json").display().to_string(),
        "--arg",
        "\"resnet\"",
        "--arg",
        "\"a100\"",
        "--arg",
        "\"imagenet\"",
        "--trace",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("inspect_model [#read_external<host>]"), "{text}");
    assert!(text.contains("4 effect(s) reached the world"), "{text}");
    assert!(text.contains("latency_ms"), "{text}");
}

#[test]
fn run_reports_a_missing_tool_instead_of_pretending() {
    let output = agent_ir(&[
        "run",
        &example("simple_agent.air"),
        "--arg",
        "\"m\"",
        "--arg",
        "\"h\"",
        "--arg",
        "\"d\"",
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("no such target"), "{}", stderr(&output));
}

#[test]
fn run_rejects_an_argument_that_is_not_json() {
    let output = agent_ir(&["run", &example("simple_agent.air"), "--arg", "resnet"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("not JSON"), "{}", stderr(&output));
}

// ------------------------------------------------------------ documentation

#[test]
fn dialects_lists_all_six() {
    let output = agent_ir(&["dialects"]);
    assert!(output.status.success());
    let text = stdout(&output);
    for dialect in ["core", "agent", "control", "tool", "memory", "observation"] {
        assert!(text.contains(dialect), "{dialect} missing from:\n{text}");
    }
    assert!(text.contains("effects: pure"), "the effect column matters:\n{text}");
}

#[test]
fn passes_lists_the_pipeline_in_order() {
    let output = agent_ir(&["passes"]);
    assert!(output.status.success());
    let text = stdout(&output);
    let dedup = text.find("tool-call-deduplication").unwrap();
    let dae = text.find("dead-action-elimination").unwrap();
    let par = text.find("parallelization").unwrap();
    assert!(dedup < dae && dae < par, "the pipeline order should be visible:\n{text}");
}

#[test]
fn the_binary_reports_its_version() {
    let output = agent_ir(&["--version"]);
    assert!(output.status.success());
    assert!(stdout(&output).contains("0.1.0"));
}
