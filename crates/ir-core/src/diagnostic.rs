//! Structured diagnostics (§14).
//!
//! Every stage of the pipeline reports failure as data, never as an opaque
//! exception: which invariant broke, which operation broke it, and what the
//! LLM builder should do about it. That is what makes the system debuggable
//! like a compiler rather than like a chain of prompts.

use crate::ids::OperationId;
use std::fmt;

/// How much a diagnostic matters.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Severity {
    /// Worth reporting; the program still runs.
    Note,
    /// Suspicious but not fatal.
    Warning,
    /// The program is rejected.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Severity::Note => "note",
            Severity::Warning => "warning",
            Severity::Error => "error",
        })
    }
}

/// A single finding, in the shape §14 prescribes.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    /// How much this finding matters.
    pub severity: Severity,
    /// The invariant (`"I2"`) or the pass that produced the finding.
    pub code: String,
    /// The offending operation, when the finding is anchored to one.
    pub operation: Option<OperationId>,
    /// What is wrong, in one sentence.
    pub message: String,
    /// What the LLM builder should change. Empty when there is nothing useful
    /// to suggest — never filled with a restatement of the message.
    pub suggestion: Option<String>,
}

impl Diagnostic {
    /// An error that rejects the program.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            code: code.into(),
            operation: None,
            message: message.into(),
            suggestion: None,
        }
    }

    /// A warning that does not reject the program.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            ..Self::error(code, message)
        }
    }

    /// An informational note.
    pub fn note(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Note,
            ..Self::error(code, message)
        }
    }

    /// Anchors the diagnostic to an operation.
    pub fn at(mut self, op: OperationId) -> Self {
        self.operation = Some(op);
        self
    }

    /// Attaches a repair suggestion.
    pub fn suggest(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Whether this finding rejects the program.
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.severity, self.code)?;
        if let Some(op) = self.operation {
            write!(f, " at {op}")?;
        }
        write!(f, ": {}", self.message)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, "\n  help: {suggestion}")?;
        }
        Ok(())
    }
}

/// A collection of findings, with the accepted/rejected verdict on top.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Diagnostics {
    entries: Vec<Diagnostic>,
}

impl Diagnostics {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a finding.
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.entries.push(diagnostic);
    }

    /// Merges another report into this one.
    pub fn extend(&mut self, other: Diagnostics) {
        self.entries.extend(other.entries);
    }

    /// Whether any finding rejects the program.
    pub fn has_errors(&self) -> bool {
        self.entries.iter().any(Diagnostic::is_error)
    }

    /// Whether nothing at all was reported.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many findings were reported.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The findings, in the order they were reported.
    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.entries.iter()
    }

    /// Only the findings that reject the program.
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.entries.iter().filter(|d| d.is_error())
    }

    /// Turns the report into `Ok(())` when nothing rejects the program.
    pub fn into_result(self) -> Result<Diagnostics, Diagnostics> {
        if self.has_errors() {
            Err(self)
        } else {
            Ok(self)
        }
    }
}

impl FromIterator<Diagnostic> for Diagnostics {
    fn from_iter<T: IntoIterator<Item = Diagnostic>>(iter: T) -> Self {
        Diagnostics {
            entries: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Diagnostics {
    type Item = Diagnostic;
    type IntoIter = std::vec::IntoIter<Diagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, entry) in self.entries.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{entry}")?;
        }
        Ok(())
    }
}
