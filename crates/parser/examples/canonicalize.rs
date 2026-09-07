//! Rewrites `.air` files into canonical form, preserving the leading comment.
//!
//! `cargo run -p agent-ir-parser --example canonicalize -- examples/*.air`
//!
//! The `agent-ir fmt` subcommand does the same thing for users; this exists so
//! the repository's own examples can be blessed without the full CLI built.

fn main() {
    let mut failed = false;
    for path in std::env::args().skip(1) {
        let source = std::fs::read_to_string(&path).expect("readable file");
        let header: Vec<&str> = source
            .lines()
            .take_while(|l| l.starts_with("//") || l.trim().is_empty())
            .collect();
        match agent_ir_parser::parse_module(&source) {
            Ok(module) => {
                let mut out = String::new();
                if !header.is_empty() {
                    out.push_str(&header.join("\n"));
                    out.push('\n');
                }
                out.push_str(&agent_ir_core::print_module(&module));
                std::fs::write(&path, out).expect("writable file");
                println!("canonicalized {path}");
            }
            Err(err) => {
                eprintln!("{path}: {err}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
