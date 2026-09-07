//! Round-trip idempotence: the phase-2 success criterion of §12.
//!
//! Two properties are checked against every example in `examples/`:
//!
//! 1. the example file is already canonical, so `print(parse(f)) == f`, and
//! 2. printing is idempotent, so `print(parse(print(m))) == print(m)`.
//!
//! The first is the strong one. It means the examples people read are exactly
//! what the compiler emits, and it fails loudly the moment printer and parser
//! drift apart.

use agent_ir_core::print_module;
use agent_ir_parser::parse_module;
use std::path::{Path, PathBuf};

fn examples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn examples() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(examples_dir())
        .expect("examples/ should exist")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension()? == "air").then_some(path)
        })
        .collect();
    found.sort();
    assert!(!found.is_empty(), "no .air examples found");
    found
}

#[test]
fn every_example_parses() {
    for path in examples() {
        let source = std::fs::read_to_string(&path).unwrap();
        if let Err(err) = parse_module(&source) {
            panic!("{}: {err}", path.display());
        }
    }
}

#[test]
fn every_example_is_already_canonical() {
    for path in examples() {
        let source = std::fs::read_to_string(&path).unwrap();
        let module = parse_module(&source).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let printed = print_module(&module);
        // Comments are not part of the IR, so compare against the file with its
        // leading comment block removed.
        let body: String = source
            .lines()
            .skip_while(|line| line.starts_with("//") || line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            printed.trim_end(),
            body.trim_end(),
            "\n{} is not canonical.\n--- printed ---\n{printed}\n--- file ---\n{body}\n",
            path.display()
        );
    }
}

#[test]
fn printing_is_idempotent() {
    for path in examples() {
        let source = std::fs::read_to_string(&path).unwrap();
        let once = print_module(&parse_module(&source).unwrap());
        let twice = print_module(&parse_module(&once).unwrap());
        assert_eq!(once, twice, "{} did not round-trip", path.display());
    }
}

#[test]
fn a_value_used_before_it_is_defined_is_rejected() {
    let err = parse_module(
        r#"
        module @bad version(0) {
          agent.func "f" {
            %x = core.cast(%y) {effect = #pure} : !core.int
          }
        }
        "#,
    )
    .unwrap_err();
    assert!(err.message.contains("before it is defined"), "{err}");
    assert!(err.message.contains("I1"), "{err}");
}

#[test]
fn an_operation_cannot_use_its_own_result() {
    let err = parse_module(
        r#"
        module @bad version(0) {
          agent.func "f" {
            %x = core.cast(%x) {effect = #pure} : !core.int
          }
        }
        "#,
    )
    .unwrap_err();
    assert!(err.message.contains("before it is defined"), "{err}");
}

#[test]
fn a_value_does_not_escape_the_region_that_defines_it() {
    let err = parse_module(
        r#"
        module @bad version(0) {
          agent.func "f" {
            control.parallel {
              %inner = core.constant {effect = #pure, value = 1} : !core.int
            } {effect = #pure}
            agent.return(%inner) {effect = #pure}
          }
        }
        "#,
    )
    .unwrap_err();
    assert!(err.message.contains("before it is defined"), "{err}");
}

#[test]
fn an_unknown_effect_is_rejected() {
    let err = parse_module(
        r#"
        module @bad version(0) {
          agent.func "f" {
            agent.return {effect = #teleport}
          }
        }
        "#,
    )
    .unwrap_err();
    assert!(err.message.contains("not one of the five effects"), "{err}");
}

#[test]
fn an_unknown_type_is_rejected() {
    let err = parse_module(
        r#"
        module @bad version(0) {
          agent.func "f" {
            %x = core.constant {effect = #pure, value = 1} : !core.quantum
          }
        }
        "#,
    )
    .unwrap_err();
    assert!(err.message.contains("not a type of §2.1"), "{err}");
}

#[test]
fn results_must_state_their_types() {
    let err = parse_module(
        r#"
        module @bad version(0) {
          agent.func "f" {
            %x = core.constant {effect = #pure, value = 1}
          }
        }
        "#,
    )
    .unwrap_err();
    assert!(err.message.contains("must state their types"), "{err}");
}

#[test]
fn capabilities_survive_a_round_trip() {
    let source = r#"module @m version(3) {
  capability @wipe scope("db") grants(irreversible, write_external) requires_approval
  capability @world scope(*) grants(read_external)

  agent.func "f" {
    agent.return {effect = #pure}
  } {effect = #pure}
}
"#;
    let module = parse_module(source).unwrap();
    assert_eq!(module.version, 3);
    let wipe = module.capabilities.get("wipe").unwrap();
    assert!(wipe.requires_approval);
    assert_eq!(wipe.grants.len(), 2);
    assert!(module.capabilities.get("world").unwrap().scope.is_any());
    assert_eq!(print_module(&module), source);
}

#[test]
fn provenance_survives_a_round_trip() {
    let source = r#"module @m version(0) {
  agent.func "f" {
    %guess = agent.action "guess"(%guess_seed) {effect = #stochastic} : !core.string provenance(%guess = llm confidence(0.4) stale)
    agent.return(%guess) {effect = #pure}
  } {effect = #pure}
}
"#;
    // `%guess_seed` is undefined on purpose in the snippet above; build a valid
    // variant instead so the parse succeeds and the provenance clause is what
    // is under test.
    let source = source.replace("(%guess_seed)", "");
    let module = parse_module(&source).unwrap();
    let value = module
        .all_values()
        .find(|v| v.name.as_deref() == Some("guess"))
        .unwrap();
    assert_eq!(value.provenance.source, agent_ir_core::Source::Llm);
    assert_eq!(value.provenance.confidence, 0.4);
    assert_eq!(value.provenance.validity, agent_ir_core::Validity::Stale);
    assert_eq!(print_module(&module), source);
}
