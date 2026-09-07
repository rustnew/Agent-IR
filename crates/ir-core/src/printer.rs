//! The canonical textual form (§3.3).
//!
//! Printing is the definition of the syntax: the parser accepts what this
//! module emits, and `print(parse(print(m))) == print(m)` is a tested property
//! (the phase-2 success criterion of §12).
//!
//! ```text
//! module @inference version(0) {
//!   capability @select_model scope("registry") grants(read_external)
//!
//!   agent.func "optimize_inference" {
//!   ^entry(%model: !tool.ref<model>):
//!     %info = agent.action "inspect_model"(%model) {effect = #pure} : !observation.observation<model>
//!     agent.return %info
//!   }
//! }
//! ```

use crate::attribute::Attribute;
use crate::ids::{BlockId, OperationId, RegionId, ValueId};
use crate::module::{Module, Operation};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// Renders a module in the canonical textual syntax.
pub fn print_module(module: &Module) -> String {
    let names = NameTable::build(module);
    let mut out = String::new();
    let _ = write!(out, "module @{} version({})", module.name, module.version);
    out.push_str(" {\n");

    for capability in module.capabilities.iter() {
        let _ = write!(out, "  capability @{}", capability.name);
        match capability.scope.name() {
            Some(scope) => {
                let _ = write!(out, " scope(\"{}\")", crate::attribute::escape(scope));
            }
            None => out.push_str(" scope(*)"),
        }
        out.push_str(" grants(");
        for (i, grant) in capability.grants.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{grant}");
        }
        out.push(')');
        if capability.requires_approval {
            out.push_str(" requires_approval");
        }
        out.push('\n');
    }
    if !module.capabilities.is_empty() {
        out.push('\n');
    }

    let mut printer = Printer {
        module,
        names: &names,
        out: &mut out,
    };
    printer.print_region_body(module.body(), 1);
    out.push_str("}\n");
    out
}

/// Renders a single operation, without its enclosing module. For diagnostics.
pub fn print_operation(module: &Module, op: OperationId) -> String {
    let names = NameTable::build(module);
    let mut out = String::new();
    let mut printer = Printer {
        module,
        names: &names,
        out: &mut out,
    };
    printer.print_op(op, 0);
    out.trim_end().to_string()
}

/// Assigns every value a name that is unique across the module.
///
/// A value's preferred name is the one it carries; collisions and unnamed
/// values fall back to a numeric suffix. Uniqueness is what makes the printed
/// form re-parseable, so it is enforced here rather than trusted.
struct NameTable {
    names: HashMap<ValueId, String>,
    blocks: HashMap<BlockId, String>,
}

impl NameTable {
    fn build(module: &Module) -> Self {
        let mut names = HashMap::new();
        let mut taken = HashSet::new();
        let mut next = 0usize;

        let mut assign = |value: &crate::module::Value, names: &mut HashMap<ValueId, String>| {
            let preferred = value.name.clone().unwrap_or_default();
            let mut candidate = if preferred.is_empty() {
                let name = next.to_string();
                next += 1;
                name
            } else {
                preferred.clone()
            };
            let mut disambiguator = 0;
            while !taken.insert(candidate.clone()) {
                disambiguator += 1;
                candidate = format!("{preferred}_{disambiguator}");
            }
            names.insert(value.id, candidate);
        };

        for value in module.all_values() {
            assign(value, &mut names);
        }

        // Block labels are numbered inside their region, not across the
        // module, so a region always starts at `^bb0`.
        let mut blocks = HashMap::new();
        for region_index in 0..module.region_count() {
            let region = module.region_by_index(region_index);
            for (position, &block) in region.blocks.iter().enumerate() {
                blocks.insert(block, format!("bb{position}"));
            }
        }

        NameTable { names, blocks }
    }

    fn value(&self, id: ValueId) -> &str {
        self.names
            .get(&id)
            .map(String::as_str)
            .unwrap_or("<detached>")
    }

    fn block(&self, id: BlockId) -> &str {
        self.blocks.get(&id).map(String::as_str).unwrap_or("bb?")
    }
}

struct Printer<'a> {
    module: &'a Module,
    names: &'a NameTable,
    out: &'a mut String,
}

impl Printer<'_> {
    fn indent(&mut self, level: usize) {
        for _ in 0..level {
            self.out.push_str("  ");
        }
    }

    /// Prints the blocks of a region, without the surrounding braces.
    fn print_region_body(&mut self, region: RegionId, level: usize) {
        let blocks = self.module.region(region).blocks.clone();
        let labelled = blocks.len() > 1
            || blocks
                .first()
                .is_some_and(|&b| !self.module.block(b).args.is_empty());

        for &block in &blocks {
            if labelled {
                self.indent(level.saturating_sub(1));
                let _ = write!(self.out, "^{}", self.names.block(block));
                let args = self.module.block(block).args.clone();
                if !args.is_empty() {
                    self.out.push('(');
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            self.out.push_str(", ");
                        }
                        let value = self.module.value(*arg);
                        let _ = write!(self.out, "%{}: {}", self.names.value(*arg), value.ty);
                    }
                    self.out.push(')');
                }
                self.out.push_str(":\n");
            }
            for op in self.module.block(block).ops.clone() {
                if self.module.op(op).erased {
                    continue;
                }
                self.print_op(op, level);
            }
        }
    }

    fn print_op(&mut self, id: OperationId, level: usize) {
        let op = self.module.op(id).clone();
        self.indent(level);

        if !op.results.is_empty() {
            for (i, result) in op.results.iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                let _ = write!(self.out, "%{}", self.names.value(*result));
            }
            self.out.push_str(" = ");
        }

        let _ = write!(self.out, "{}", op.name);

        if let Some(literal) = &op.literal {
            let _ = write!(self.out, " \"{}\"", crate::attribute::escape(literal));
        }

        if !op.operands.is_empty() {
            self.out.push('(');
            for (i, operand) in op.operands.iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                let _ = write!(self.out, "%{}", self.names.value(*operand));
            }
            self.out.push(')');
        }

        for &region in &op.regions {
            self.out.push_str(" {\n");
            self.print_region_body(region, level + 1);
            self.indent(level);
            self.out.push('}');
        }

        self.print_attributes(&op);

        if !op.results.is_empty() {
            self.out.push_str(" : ");
            if op.results.len() == 1 {
                let _ = write!(self.out, "{}", self.module.value(op.results[0]).ty);
            } else {
                self.out.push('(');
                for (i, result) in op.results.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    let _ = write!(self.out, "{}", self.module.value(*result).ty);
                }
                self.out.push(')');
            }
        }

        self.print_provenance(&op);
        self.out.push('\n');
    }

    /// Prints the attribute dictionary, with the declared effect folded in.
    ///
    /// The effect lives on the operation rather than in the dictionary, but it
    /// prints as `effect = #...` because that is how §3.3 writes it and how an
    /// LLM builder is expected to emit it.
    fn print_attributes(&mut self, op: &Operation) {
        self.out.push_str(" {");
        let mut first = true;
        let entry = |out: &mut String, first: &mut bool| {
            if *first {
                *first = false;
            } else {
                out.push_str(", ");
            }
        };

        entry(self.out, &mut first);
        let _ = write!(self.out, "effect = {}", op.effect);

        for (key, value) in &op.attributes {
            if key == "effect" {
                continue;
            }
            entry(self.out, &mut first);
            let _ = write!(self.out, "{key} = {value}");
        }
        self.out.push('}');
    }

    /// Prints provenance for the results that carry something other than the
    /// default, so a printed module keeps everything the verifier needs.
    fn print_provenance(&mut self, op: &Operation) {
        let interesting: Vec<&ValueId> = op
            .results
            .iter()
            .filter(|&&id| {
                let p = &self.module.value(id).provenance;
                p.source != crate::provenance::Source::Derived
                    || p.confidence != 1.0
                    || p.validity != crate::provenance::Validity::Valid
            })
            .collect();
        if interesting.is_empty() {
            return;
        }
        self.out.push_str(" provenance(");
        for (i, id) in interesting.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let value = self.module.value(**id);
            let p = &value.provenance;
            let _ = write!(self.out, "%{} = {}", self.names.value(**id), p.source);
            if p.confidence != 1.0 {
                let _ = write!(self.out, " confidence({})", FloatLiteral(p.confidence));
            }
            if p.validity != crate::provenance::Validity::Valid {
                let _ = write!(self.out, " {}", p.validity);
            }
        }
        self.out.push(')');
    }
}

/// Formats a float the way [`Attribute`] does, so both round-trip identically.
pub(crate) struct FloatLiteral(pub f64);

impl std::fmt::Display for FloatLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::attribute::write_float(f, self.0)
    }
}

impl std::fmt::Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&print_module(self))
    }
}

/// Re-exported so callers can format an attribute the same way the printer does.
pub fn print_attribute(attribute: &Attribute) -> String {
    attribute.to_string()
}
