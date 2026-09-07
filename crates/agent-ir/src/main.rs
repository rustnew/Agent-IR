//! `agent-ir` — the command line driver.
//!
//! One subcommand per stage of §4, so each rejection point can be looked at on
//! its own:
//!
//! ```text
//! agent-ir fmt      canonicalize a program
//! agent-ir verify   the two rejection points of §4
//! agent-ir opt      run the §5 passes and show what they did
//! agent-ir plan     lower and schedule (§6.4, §15)
//! agent-ir run      execute against a stubbed environment (§8)
//! agent-ir dialects what the v0.1 dialects contain (§3.2)
//! agent-ir passes   what each pass does and when it may fire (§5)
//! ```

use agent_ir::core::{Diagnostics, Module};
use agent_ir::dialects::Registry;
use agent_ir::lowering::{GenericRuntime, Scheduler};
use agent_ir::passes::PassManager;
use agent_ir::runtime::{
    Executor, InMemoryCheckpointStore, InMemoryEventLog, EventLog as _, RecordingEnvironment,
    Value,
};
use agent_ir::verifier::Verifier;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "agent-ir",
    version,
    about = "A compilation infrastructure for agentic systems",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Rewrites a program into canonical form.
    Fmt {
        /// The `.air` files to format.
        files: Vec<PathBuf>,
        /// Report which files would change instead of rewriting them.
        #[arg(long)]
        check: bool,
    },
    /// Runs both verification phases of §4.
    Verify {
        /// The program to check.
        file: PathBuf,
        /// Stop after the static phase, without asking about capabilities.
        #[arg(long)]
        static_only: bool,
        /// Emit the diagnostics as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Runs the optimization passes and prints the result.
    Opt {
        /// The program to optimize.
        file: PathBuf,
        /// Print the pass report instead of the optimized program.
        #[arg(long)]
        report: bool,
    },
    /// Lowers and schedules one function.
    Plan {
        /// The program to schedule.
        file: PathBuf,
        /// Which function; defaults to the only one, if there is only one.
        #[arg(long)]
        function: Option<String>,
        /// Skip the optimization passes.
        #[arg(long)]
        no_opt: bool,
        /// Emit the plan as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Executes a function against a stubbed environment.
    Run {
        /// The program to run.
        file: PathBuf,
        /// Which function; defaults to the only one, if there is only one.
        #[arg(long)]
        function: Option<String>,
        /// A JSON object mapping each tool name to the value it should return.
        #[arg(long)]
        tools: Option<PathBuf>,
        /// One JSON argument per function parameter, in order.
        #[arg(long = "arg")]
        args: Vec<String>,
        /// Print the event log after the run.
        #[arg(long)]
        trace: bool,
    },
    /// Lists the operations of the v0.1 dialects.
    Dialects,
    /// Lists the optimization passes and their validity conditions.
    Passes,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Fmt { files, check } => fmt(files, check),
        Command::Verify { file, static_only, json } => verify(file, static_only, json),
        Command::Opt { file, report } => opt(file, report),
        Command::Plan { file, function, no_opt, json } => plan(file, function, no_opt, json),
        Command::Run { file, function, tools, args, trace } => {
            execute(file, function, tools, args, trace)
        }
        Command::Dialects => {
            dialects();
            Ok(())
        }
        Command::Passes => {
            passes();
            Ok(())
        }
    }
}

// ------------------------------------------------------------------ commands

fn fmt(files: Vec<PathBuf>, check: bool) -> Result<(), String> {
    if files.is_empty() {
        return Err("agent-ir fmt needs at least one file".into());
    }
    let mut would_change = Vec::new();
    for path in files {
        let source = read(&path)?;
        let module = parse(&source, &path)?;
        let header: Vec<&str> = source
            .lines()
            .take_while(|l| l.starts_with("//") || l.trim().is_empty())
            .collect();
        let mut formatted = String::new();
        if !header.is_empty() {
            formatted.push_str(&header.join("\n"));
            formatted.push('\n');
        }
        formatted.push_str(&agent_ir::core::print_module(&module));

        if formatted == source {
            continue;
        }
        if check {
            would_change.push(path);
        } else {
            std::fs::write(&path, formatted)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            println!("formatted {}", path.display());
        }
    }
    if !would_change.is_empty() {
        let names: Vec<String> = would_change.iter().map(|p| p.display().to_string()).collect();
        return Err(format!("not canonical:\n  {}", names.join("\n  ")));
    }
    Ok(())
}

fn verify(path: PathBuf, static_only: bool, json: bool) -> Result<(), String> {
    let source = read(&path)?;
    let module = parse(&source, &path)?;
    let verifier = Verifier::new();
    let report = if static_only {
        verifier.verify(&module)
    } else {
        verifier.verify_all(&module)
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else if report.is_empty() {
        println!("{}: accepted", path.display());
    } else {
        println!("{report}");
    }

    if report.has_errors() {
        Err(format!(
            "{}: rejected, {} error(s)",
            path.display(),
            report.errors().count()
        ))
    } else {
        Ok(())
    }
}

fn opt(path: PathBuf, show_report: bool) -> Result<(), String> {
    let source = read(&path)?;
    let mut module = parse(&source, &path)?;
    reject_if_invalid(&path, &Verifier::new().verify_all(&module))?;

    let report = PassManager::default_pipeline().run(&mut module);

    reject_if_invalid(&path, &Verifier::new().verify_all(&module))
        .map_err(|e| format!("a pass produced an invalid module — this is a compiler bug\n{e}"))?;

    if show_report {
        print!("{report}");
        for pass in report.reports.iter().filter(|r| r.changed) {
            for note in pass.notes.iter() {
                println!("    {note}");
            }
        }
    } else {
        print!("{module}");
    }
    Ok(())
}

fn plan(
    path: PathBuf,
    function: Option<String>,
    no_opt: bool,
    json: bool,
) -> Result<(), String> {
    let source = read(&path)?;
    let mut module = parse(&source, &path)?;
    reject_if_invalid(&path, &Verifier::new().verify_all(&module))?;
    if !no_opt {
        PassManager::default_pipeline().run(&mut module);
    }
    let name = pick_function(&module, function)?;
    let plan = Scheduler::new(GenericRuntime)
        .schedule(&module, &name)
        .map_err(|report| format!("{report}"))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("plan for @{} :: {} (backend: {})", plan.module, plan.function, plan.backend);
    println!("estimated: {}", plan.estimated);
    println!();
    print_plan(&plan.plan, 0);
    for note in &plan.notes {
        println!("\n{note}");
    }
    Ok(())
}

fn print_plan(plan: &agent_ir::lowering::Plan, depth: usize) {
    let pad = "  ".repeat(depth);
    for (index, batch) in plan.batches.iter().enumerate() {
        let concurrency = if batch.len() > 1 {
            format!(" ({} concurrent)", batch.len())
        } else {
            String::new()
        };
        println!("{pad}batch {index}{concurrency}");
        for step in batch {
            let literal = step
                .literal
                .as_deref()
                .map(|l| format!(" \"{l}\""))
                .unwrap_or_default();
            println!("{pad}  {}{literal}  →  {}", step.name, step.target);
            if let Some(key) = &step.idempotency_key {
                println!("{pad}    idempotency: {key}");
            }
            match step.control.as_deref() {
                Some(agent_ir::lowering::Control::If { then, otherwise }) => {
                    println!("{pad}    then:");
                    print_plan(then, depth + 3);
                    if let Some(otherwise) = otherwise {
                        println!("{pad}    else:");
                        print_plan(otherwise, depth + 3);
                    }
                }
                Some(agent_ir::lowering::Control::Loop { max_iterations, body, .. }) => {
                    println!("{pad}    body (up to {max_iterations} iterations):");
                    print_plan(body, depth + 3);
                }
                Some(agent_ir::lowering::Control::While { condition, body, .. }) => {
                    println!("{pad}    condition:");
                    print_plan(condition, depth + 3);
                    println!("{pad}    body:");
                    print_plan(body, depth + 3);
                }
                Some(agent_ir::lowering::Control::Parallel { body }) => {
                    println!("{pad}    concurrently:");
                    print_plan(body, depth + 3);
                }
                None => {}
            }
        }
    }
}

fn execute(
    path: PathBuf,
    function: Option<String>,
    tools: Option<PathBuf>,
    args: Vec<String>,
    trace: bool,
) -> Result<(), String> {
    let source = read(&path)?;
    let mut module = parse(&source, &path)?;
    reject_if_invalid(&path, &Verifier::new().verify_all(&module))?;
    PassManager::default_pipeline().run(&mut module);

    let name = pick_function(&module, function)?;
    let plan = Scheduler::new(GenericRuntime)
        .schedule(&module, &name)
        .map_err(|report| format!("{report}"))?;

    let mut env = RecordingEnvironment::new();
    if let Some(tools) = tools {
        let text = read(&tools)?;
        let table: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text)
            .map_err(|e| format!("{}: {e}", tools.display()))?;
        for (name, value) in table {
            env = env.returning(&name, Value::from_json(value));
        }
    }

    let mut arguments = Vec::new();
    for (index, arg) in args.iter().enumerate() {
        let json: serde_json::Value = serde_json::from_str(arg)
            .map_err(|e| format!("--arg #{index} is not JSON: {e}"))?;
        arguments.push(Value::from_json(json));
    }

    let mut log = InMemoryEventLog::new();
    let mut checkpoints = InMemoryCheckpointStore::new();
    let outcome = Executor::new(&mut env, &mut log, &mut checkpoints)
        .run(&plan, arguments)
        .map_err(|e| {
            let mut message = format!("execution stopped: {e}");
            if trace {
                message.push_str(&format!("\n\n{log}"));
            }
            message
        })?;

    if trace {
        print!("{log}");
        println!();
    }
    println!("result: {}", outcome.result);
    println!(
        "{} effect(s) reached the world, {} replayed from the ledger",
        outcome.effects, outcome.replayed
    );
    if !outcome.state.rejected.is_empty() {
        println!("rejected: {} candidate(s)", outcome.state.rejected.len());
    }
    let _ = log.len();
    Ok(())
}

fn dialects() {
    let registry = Registry::v0_1();
    for dialect in agent_ir::dialects::names::ALL {
        println!("{dialect}");
        for signature in registry.dialect(dialect) {
            let effects: Vec<String> = signature
                .allowed_effects
                .iter()
                .map(ToString::to_string)
                .collect();
            let effects = if effects.is_empty() {
                "any".to_string()
            } else {
                effects.join(" | ")
            };
            println!("  {:<24} {}", signature.name.name, signature.summary);
            println!(
                "  {:<24} operands: {}, results: {}, effects: {effects}",
                "",
                signature.operands.describe(),
                signature.results.describe()
            );
        }
        println!();
    }
}

fn passes() {
    for pass in PassManager::default_pipeline().passes() {
        println!("{}", pass.name());
        println!("  {}", pass.description());
    }
}

// ------------------------------------------------------------------- helpers

fn read(path: &PathBuf) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse(source: &str, path: &PathBuf) -> Result<Module, String> {
    agent_ir::parser::parse_module(source).map_err(|e| format!("{}: {e}", path.display()))
}

fn reject_if_invalid(path: &PathBuf, report: &Diagnostics) -> Result<(), String> {
    if report.has_errors() {
        return Err(format!("{}: rejected\n{report}", path.display()));
    }
    Ok(())
}

fn pick_function(module: &Module, requested: Option<String>) -> Result<String, String> {
    let names: Vec<String> = module
        .top_level()
        .into_iter()
        .filter(|&op| module.op(op).name.is("agent", "func"))
        .filter_map(|op| module.op(op).literal.clone())
        .collect();

    match requested {
        Some(name) if names.contains(&name) => Ok(name),
        Some(name) => Err(format!(
            "no function `{name}`; this module has: {}",
            names.join(", ")
        )),
        None if names.len() == 1 => Ok(names[0].clone()),
        None if names.is_empty() => Err("this module defines no functions".into()),
        None => Err(format!(
            "--function is required; this module has: {}",
            names.join(", ")
        )),
    }
}
