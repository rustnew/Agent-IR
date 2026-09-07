//! The specification and the implementation must not drift apart.
//!
//! README.md is the specification and the project's front page at once. Every
//! program it prints is therefore checked here against the compiler that is
//! supposed to accept it — a specification whose own worked example does not
//! compile is a pitch, not a specification.

use agent_ir::core::print_module;
use agent_ir::dialects::Registry;
use agent_ir::parser::parse_module;
use agent_ir::verifier::Verifier;
use std::path::{Path, PathBuf};

fn repo(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn readme() -> String {
    std::fs::read_to_string(repo("README.md")).expect("README.md should be readable")
}

/// Every fenced block in the README tagged with `language`.
fn code_blocks(source: &str, language: &str) -> Vec<String> {
    let fence = format!("```{language}");
    let mut blocks = Vec::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        if line.trim() != fence {
            continue;
        }
        let mut block = Vec::new();
        for line in lines.by_ref() {
            if line.trim() == "```" {
                break;
            }
            block.push(line);
        }
        blocks.push(block.join("\n"));
    }
    blocks
}

#[test]
fn the_worked_example_in_section_3_3_is_the_checked_in_program() {
    let blocks = code_blocks(&readme(), "mlir");
    assert_eq!(blocks.len(), 2, "§3.3 and §7 each hold one `mlir` block");

    let example = std::fs::read_to_string(repo("examples/optimize_inference.air")).unwrap();
    let body: String = example
        .lines()
        .skip_while(|l| l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(
        blocks[0].trim(),
        body.trim(),
        "§3.3 has drifted from examples/optimize_inference.air"
    );
}

#[test]
fn every_program_the_specification_prints_actually_compiles() {
    for (index, block) in code_blocks(&readme(), "mlir").iter().enumerate() {
        // §7's block is a fragment illustrating a rejection, not a module.
        if !block.trim_start().starts_with("module") {
            continue;
        }
        let module = parse_module(block)
            .unwrap_or_else(|e| panic!("mlir block #{index} does not parse: {e}"));
        let report = Verifier::new().verify_all(&module);
        assert!(
            !report.has_errors(),
            "mlir block #{index} does not verify:\n{report}"
        );
        assert_eq!(
            print_module(&module).trim(),
            block.trim(),
            "mlir block #{index} is not canonical"
        );
    }
}

#[test]
fn the_dialect_list_in_section_3_2_matches_the_registry() {
    let source = readme();
    let start = source
        .find("### 3.2 v0.1 dialects")
        .expect("§3.2 should exist");
    let section = &source[start..start + 1200];
    let registry = Registry::v0_1();

    for dialect in agent_ir::dialects::names::ALL {
        for signature in registry.dialect(dialect) {
            let listed = section
                .lines()
                .find(|line| line.trim_start().starts_with(&format!("{dialect} ")))
                .unwrap_or("");
            assert!(
                listed.contains(&signature.name.name),
                "§3.2 does not list `{}`; the line reads: {listed}",
                signature.name
            );
        }
    }
}

#[test]
fn the_specification_claims_no_measured_performance() {
    // §0 rule 1. A number followed by a speedup unit anywhere in this document
    // would be a claim the benchmark of §9.4 has not earned.
    let banned = ["x faster", "% faster", "speedup", "× faster"];
    let source = readme().to_lowercase();
    for phrase in banned {
        assert!(
            !source.contains(phrase),
            "§0 forbids an unmeasured performance claim, but the text contains `{phrase}`"
        );
    }
}

#[test]
fn every_section_the_code_cites_exists() {
    // The crates reference the specification by section number throughout. A
    // dangling reference is a small lie that compounds.
    let source = readme();
    let sections: Vec<String> = source
        .lines()
        .filter_map(|l| l.strip_prefix("## ").or_else(|| l.strip_prefix("### ")))
        .filter_map(|l| l.split_whitespace().next())
        .map(|n| n.trim_end_matches('.').to_string())
        .collect();

    let mut cited = std::collections::BTreeSet::new();
    for entry in walk(&repo("crates")) {
        let text = std::fs::read_to_string(&entry).unwrap_or_default();
        let mut rest = text.as_str();
        while let Some(at) = rest.find('§') {
            rest = &rest[at + '§'.len_utf8()..];
            let number: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let number = number.trim_end_matches('.').to_string();
            if !number.is_empty() {
                cited.insert(number);
            }
        }
    }

    assert!(!cited.is_empty(), "the code should cite the specification");
    for reference in &cited {
        assert!(
            sections.contains(reference),
            "the code cites §{reference}, which this document does not define"
        );
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            found.push(path);
        }
    }
    found
}
