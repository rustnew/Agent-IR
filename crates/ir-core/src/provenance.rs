//! Value provenance (§2.4).
//!
//! Provenance is what lets the system answer "why did this action happen", and
//! what lets invariant I4 refuse to feed a low-confidence LLM guess into an
//! operation that touches the world.

use crate::ids::OperationId;
use std::fmt;

/// Where a value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Source {
    /// Proposed by the LLM builder: unreliable until verified.
    Llm,
    /// Returned by a tool: reliable up to the tool's own truthfulness.
    Tool,
    /// Supplied by the user.
    User,
    /// Read back from the memory store.
    Memory,
    /// Computed from other values by a pure operation.
    Derived,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Source::Llm => "llm",
            Source::Tool => "tool",
            Source::User => "user",
            Source::Memory => "memory",
            Source::Derived => "derived",
        })
    }
}

impl std::str::FromStr for Source {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "llm" => Source::Llm,
            "tool" => Source::Tool,
            "user" => Source::User,
            "memory" => Source::Memory,
            "derived" => Source::Derived,
            _ => return Err(()),
        })
    }
}

/// Whether a value is still to be trusted (§8.5).
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum Validity {
    /// Still trustworthy.
    #[default]
    Valid,
    /// Past its freshness window; must be re-read before use.
    Stale,
    /// Known to be wrong.
    Invalid,
    /// Moved out of the active window (§8.1).
    Archived,
}

impl fmt::Display for Validity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Validity::Valid => "valid",
            Validity::Stale => "stale",
            Validity::Invalid => "invalid",
            Validity::Archived => "archived",
        })
    }
}

impl std::str::FromStr for Validity {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "valid" => Validity::Valid,
            "stale" => Validity::Stale,
            "invalid" => Validity::Invalid,
            "archived" => Validity::Archived,
            _ => return Err(()),
        })
    }
}

/// The origin record every value carries (§2.4).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    /// The operation that produced the value, if any (block arguments have none).
    pub producer: Option<OperationId>,
    /// Who produced the value.
    pub source: Source,
    /// Logical timestamp: the IR version in which the value appeared.
    pub timestamp: u64,
    /// `1.0` for reliable tool data, below that for an LLM inference.
    pub confidence: f64,
    /// Whether the value may still be relied upon.
    pub validity: Validity,
}

impl Provenance {
    /// Provenance for a value derived by a pure operation: fully trusted.
    pub fn derived() -> Self {
        Provenance {
            producer: None,
            source: Source::Derived,
            timestamp: 0,
            confidence: 1.0,
            validity: Validity::Valid,
        }
    }

    /// Provenance for a value returned by a tool.
    pub fn from_tool() -> Self {
        Provenance {
            source: Source::Tool,
            ..Provenance::derived()
        }
    }

    /// Provenance for a value supplied by the user.
    pub fn from_user() -> Self {
        Provenance {
            source: Source::User,
            ..Provenance::derived()
        }
    }

    /// Provenance for a value proposed by the LLM, with its confidence.
    pub fn from_llm(confidence: f64) -> Self {
        Provenance {
            source: Source::Llm,
            confidence,
            ..Provenance::derived()
        }
    }

    /// Whether the value needs an explicit check before feeding a non-`Pure`
    /// operation, given the configured confidence threshold (invariant I4).
    pub fn needs_verification(&self, threshold: f64) -> bool {
        self.source == Source::Llm && self.confidence < threshold
    }

    /// Whether the value may still be relied upon.
    pub fn is_usable(&self) -> bool {
        matches!(self.validity, Validity::Valid)
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Provenance::derived()
    }
}
