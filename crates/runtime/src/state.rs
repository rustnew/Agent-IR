//! Execution state, the idempotency ledger and checkpoints (§8.1–§8.3, §8.6).
//!
//! §8.1 asks for a strict separation between what the agent is doing now, what
//! actually happened, and what survives a crash. [`ExecutionState`] is the
//! first; the event log is the second; a [`Checkpoint`] is what carries the
//! third across a restart.
//!
//! The important piece is the [`Ledger`]. It records, for each idempotency key,
//! the result of the effect that key guarded. On resume the plan is re-run from
//! the top, and every non-replayable step whose key is already in the ledger
//! returns its recorded result instead of touching the world again. That is
//! what makes replay-based recovery *sound* rather than merely convenient — and
//! it works only because the effect system already told us which steps are
//! replayable and which are not.

use crate::value::Value;
use agent_ir_core::ValueId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Results of effects that must not happen twice, keyed by idempotency key.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    entries: BTreeMap<String, Value>,
}

impl Ledger {
    /// An empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that the effect behind `key` completed with `result`.
    pub fn record(&mut self, key: impl Into<String>, result: Value) {
        self.entries.insert(key.into(), result);
    }

    /// The recorded result for a key, if the effect already happened.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Whether the effect behind a key already happened.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// How many completed effects are recorded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has completed yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every key, in order.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }
}

/// What the agent is doing right now (§8.1, "operational state").
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionState {
    /// The value bound to each SSA value so far.
    pub values: BTreeMap<ValueId, Value>,
    /// Effects that already happened, and their results.
    pub ledger: Ledger,
    /// Candidates the program explicitly rejected, for the audit trail.
    pub rejected: Vec<Value>,
    /// How many times an identical action has repeated, for §8.4.
    pub repeats: BTreeMap<String, usize>,
}

impl ExecutionState {
    /// An empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds a value.
    pub fn bind(&mut self, id: ValueId, value: Value) {
        self.values.insert(id, value);
    }

    /// Reads a bound value.
    pub fn get(&self, id: ValueId) -> Option<&Value> {
        self.values.get(&id)
    }

    /// Reads a bound value, or `Null` when the program never produced one.
    pub fn get_or_null(&self, id: ValueId) -> Value {
        self.values.get(&id).cloned().unwrap_or(Value::Null)
    }

    /// Counts one more occurrence of an action signature, returning the new
    /// total (§8.4).
    pub fn count_repeat(&mut self, signature: String) -> usize {
        let counter = self.repeats.entry(signature).or_insert(0);
        *counter += 1;
        *counter
    }
}

/// A consistent snapshot of an execution (§8.2).
///
/// Snapshots are whole or absent: a partial checkpoint would let recovery
/// resume from a state that never existed, which is the risk §5 lists against
/// *Checkpoint Optimization*.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Which checkpoint this is, counting from one.
    pub sequence: u64,
    /// The last event-log entry included in the snapshot.
    pub log_position: u64,
    /// The IR version this execution was compiled from (§1.2).
    pub version: u64,
    /// The state at the moment of the snapshot.
    pub state: ExecutionState,
}

/// Where checkpoints live.
pub trait CheckpointStore {
    /// Stores a snapshot.
    fn save(&mut self, checkpoint: Checkpoint);

    /// The most recent snapshot, if there is one.
    fn latest(&self) -> Option<&Checkpoint>;

    /// Every snapshot, oldest first.
    fn all(&self) -> &[Checkpoint];
}

/// An in-memory checkpoint store.
#[derive(Clone, Debug, Default)]
pub struct InMemoryCheckpointStore {
    checkpoints: Vec<Checkpoint>,
}

impl InMemoryCheckpointStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl CheckpointStore for InMemoryCheckpointStore {
    fn save(&mut self, checkpoint: Checkpoint) {
        self.checkpoints.push(checkpoint);
    }

    fn latest(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    fn all(&self) -> &[Checkpoint] {
        &self.checkpoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ledger_remembers_one_result_per_key() {
        let mut ledger = Ledger::new();
        ledger.record("k", Value::Int(1));
        ledger.record("k", Value::Int(2));
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.get("k"), Some(&Value::Int(2)));
        assert!(!ledger.contains("other"));
    }

    #[test]
    fn repeat_counting_is_per_signature() {
        let mut state = ExecutionState::new();
        assert_eq!(state.count_repeat("a".into()), 1);
        assert_eq!(state.count_repeat("a".into()), 2);
        assert_eq!(state.count_repeat("b".into()), 1);
    }

    #[test]
    fn the_store_hands_back_the_newest_snapshot() {
        let mut store = InMemoryCheckpointStore::new();
        for sequence in 1..=3 {
            store.save(Checkpoint {
                sequence,
                log_position: sequence * 10,
                version: 0,
                state: ExecutionState::new(),
            });
        }
        assert_eq!(store.latest().unwrap().sequence, 3);
        assert_eq!(store.all().len(), 3);
    }

    #[test]
    fn a_checkpoint_round_trips_through_json() {
        let mut state = ExecutionState::new();
        state.ledger.record("pay:1", Value::Str("receipt".into()));
        let checkpoint =
            Checkpoint { sequence: 1, log_position: 9, version: 2, state };
        let json = serde_json::to_string(&checkpoint).unwrap();
        assert_eq!(
            serde_json::from_str::<Checkpoint>(&json).unwrap(),
            checkpoint
        );
    }
}
