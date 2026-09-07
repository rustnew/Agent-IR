//! The append-only event log of §8.1 and §8.2.
//!
//! The log is the source of truth for what actually happened. It is what
//! recovery replays, and it is the audit trail §6 asks for: every entry points
//! back at the IR operation that caused it, so "why did this action happen" is
//! answerable after the fact.
//!
//! Append-only is a property, not a convention: [`EventLog`] has no method that
//! removes or rewrites an entry.

use crate::value::Value;
use agent_ir_core::OperationId;
use serde::{Deserialize, Serialize};
use std::fmt;

/// What happened.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventKind {
    /// Execution of a function began.
    Started {
        /// The function.
        function: String,
        /// The IR version it was compiled from.
        version: u64,
    },
    /// A step reached the world and came back.
    Effect {
        /// What was invoked.
        target: String,
        /// The declared effect.
        effect: String,
        /// The key that guards re-execution, when there is one.
        idempotency_key: Option<String>,
        /// What came back.
        result: Value,
    },
    /// A step was skipped because its idempotency key was already recorded
    /// (§8.3). This is the entry that proves an effect did not happen twice.
    EffectReplayed {
        /// What would have been invoked.
        target: String,
        /// The key that matched.
        idempotency_key: String,
        /// The result that was reused.
        result: Value,
    },
    /// A step failed.
    Failed {
        /// What was invoked.
        target: String,
        /// Why it failed.
        error: String,
    },
    /// A loop guard fired (§8.4).
    LoopGuard {
        /// The repeated action.
        target: String,
        /// How many times it repeated identically.
        repeats: usize,
        /// What the policy decided to do.
        action: String,
    },
    /// State was checkpointed (§8.2).
    Checkpoint {
        /// Which checkpoint.
        sequence: u64,
        /// How many completed effects it carries.
        effects: usize,
    },
    /// Execution resumed from a checkpoint (§8.6).
    Resumed {
        /// The checkpoint restored.
        sequence: u64,
    },
    /// Execution finished.
    Finished {
        /// The value the function returned.
        result: Value,
    },
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventKind::Started { function, version } => {
                write!(f, "started {function} from IR{version}")
            }
            EventKind::Effect { target, effect, .. } => write!(f, "{target} [{effect}]"),
            EventKind::EffectReplayed { target, .. } => write!(f, "{target} [replayed]"),
            EventKind::Failed { target, error } => write!(f, "{target} failed: {error}"),
            EventKind::LoopGuard {
                target,
                repeats,
                action,
            } => {
                write!(
                    f,
                    "loop guard on {target} after {repeats} repeats: {action}"
                )
            }
            EventKind::Checkpoint { sequence, effects } => {
                write!(f, "checkpoint {sequence} ({effects} effect(s))")
            }
            EventKind::Resumed { sequence } => write!(f, "resumed from checkpoint {sequence}"),
            EventKind::Finished { result } => write!(f, "finished: {result}"),
        }
    }
}

/// One entry in the log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Position in the log, from zero.
    pub sequence: u64,
    /// The IR operation responsible, when there is one.
    pub op: Option<OperationId>,
    /// What happened.
    pub kind: EventKind,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:<4}", self.sequence)?;
        match self.op {
            Some(op) => write!(f, " {op:>6}  ")?,
            None => write!(f, "         ")?,
        }
        write!(f, "{}", self.kind)
    }
}

/// An append-only record of what happened.
pub trait EventLog {
    /// Appends an entry and returns its sequence number.
    fn append(&mut self, op: Option<OperationId>, kind: EventKind) -> u64;

    /// Every entry so far, oldest first.
    fn entries(&self) -> &[Event];

    /// How many entries the log holds.
    fn len(&self) -> usize {
        self.entries().len()
    }

    /// Whether nothing has been recorded.
    fn is_empty(&self) -> bool {
        self.entries().is_empty()
    }

    /// The entries after a given sequence number, for replay from a checkpoint.
    fn since(&self, sequence: u64) -> Vec<&Event> {
        self.entries()
            .iter()
            .filter(|event| event.sequence > sequence)
            .collect()
    }
}

/// An in-memory log. The default, and what the tests assert against.
#[derive(Clone, Debug, Default)]
pub struct InMemoryEventLog {
    events: Vec<Event>,
}

impl InMemoryEventLog {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// The log rendered one entry per line, as JSON — the durable form of §8.2.
    pub fn to_jsonl(&self) -> String {
        self.events
            .iter()
            .map(|event| serde_json::to_string(event).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Reads a log back from its JSON-lines form.
    pub fn from_jsonl(text: &str) -> Result<Self, serde_json::Error> {
        let mut events = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            events.push(serde_json::from_str(line)?);
        }
        Ok(InMemoryEventLog { events })
    }
}

impl EventLog for InMemoryEventLog {
    fn append(&mut self, op: Option<OperationId>, kind: EventKind) -> u64 {
        let sequence = self.events.len() as u64;
        self.events.push(Event { sequence, op, kind });
        sequence
    }

    fn entries(&self) -> &[Event] {
        &self.events
    }
}

impl fmt::Display for InMemoryEventLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for event in &self.events {
            writeln!(f, "{event}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_numbers_are_dense_and_ordered() {
        let mut log = InMemoryEventLog::new();
        for i in 0..5 {
            let sequence = log.append(
                None,
                EventKind::Checkpoint {
                    sequence: i,
                    effects: 0,
                },
            );
            assert_eq!(sequence, i);
        }
        assert_eq!(log.len(), 5);
    }

    #[test]
    fn since_returns_only_what_came_after() {
        let mut log = InMemoryEventLog::new();
        for _ in 0..4 {
            log.append(
                None,
                EventKind::Finished {
                    result: Value::Null,
                },
            );
        }
        assert_eq!(log.since(1).len(), 2);
    }

    #[test]
    fn the_log_round_trips_through_json_lines() {
        let mut log = InMemoryEventLog::new();
        log.append(
            None,
            EventKind::Started {
                function: "f".into(),
                version: 3,
            },
        );
        log.append(
            None,
            EventKind::Effect {
                target: "fetch".into(),
                effect: "#read_external<web>".into(),
                idempotency_key: None,
                result: Value::Int(7),
            },
        );
        let text = log.to_jsonl();
        let restored = InMemoryEventLog::from_jsonl(&text).unwrap();
        assert_eq!(restored.entries(), log.entries());
    }
}
