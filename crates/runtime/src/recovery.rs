//! Crash recovery (§8.6, §16).
//!
//! ```text
//! CRASH → Recovery Manager → last valid checkpoint → state validation → resume
//! ```
//!
//! Recovery here is replay, not rewind. The plan is re-run from the top with
//! the ledger restored from the last checkpoint; every non-replayable step
//! whose key is already recorded returns that result instead of touching the
//! world, and every replayable step simply runs again — which is safe, because
//! that is exactly what `Pure` and `ReadExternal` mean.
//!
//! So the effect system is not only what makes the §5 passes decidable. It is
//! also what makes recovery correct: without it there would be no principled
//! way to know which steps may be re-run and which must not.

use crate::event::{EventKind, EventLog};
use crate::state::{CheckpointStore, ExecutionState, Ledger};

/// What a resume decided to start from.
#[derive(Clone, Debug, PartialEq)]
pub struct Resumption {
    /// The state to hand the executor.
    pub state: ExecutionState,
    /// The checkpoint it came from, or `None` when there was none and the run
    /// starts over.
    pub from_checkpoint: Option<u64>,
    /// How many completed effects were recovered from the log *after* the
    /// checkpoint — the writes whose acknowledgement the crash swallowed.
    pub recovered_from_log: usize,
}

/// Rebuilds a resumable state out of a checkpoint store and an event log.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecoveryManager;

impl RecoveryManager {
    /// A recovery manager.
    pub fn new() -> Self {
        RecoveryManager
    }

    /// The state to resume from.
    ///
    /// The checkpoint supplies the bulk of it. The log tail supplies the rest:
    /// effects that completed after the last checkpoint are still in the log,
    /// and folding them back into the ledger is what stops them happening a
    /// second time.
    pub fn resume<L: EventLog, C: CheckpointStore>(
        &self,
        log: &L,
        checkpoints: &C,
    ) -> Resumption {
        let (mut state, position, from_checkpoint) = match checkpoints.latest() {
            Some(checkpoint) => (
                checkpoint.state.clone(),
                checkpoint.log_position,
                Some(checkpoint.sequence),
            ),
            None => (ExecutionState::new(), 0, None),
        };

        let recovered = fold_log_tail(log, position, &mut state.ledger);
        Resumption { state, from_checkpoint, recovered_from_log: recovered }
    }

    /// Rebuilds a ledger from the log alone, with no checkpoint at all.
    ///
    /// The log is the source of truth; a checkpoint is only an optimization
    /// that saves replaying it from the beginning.
    pub fn ledger_from_log<L: EventLog>(&self, log: &L) -> Ledger {
        let mut ledger = Ledger::new();
        fold_log_tail(log, 0, &mut ledger);
        ledger
    }

    /// Records that a resume happened, so the audit trail shows the seam.
    pub fn note_resumption<L: EventLog>(&self, log: &mut L, resumption: &Resumption) {
        if let Some(sequence) = resumption.from_checkpoint {
            log.append(None, EventKind::Resumed { sequence });
        }
    }
}

/// Folds every completed effect after `position` back into the ledger.
fn fold_log_tail<L: EventLog>(log: &L, position: u64, ledger: &mut Ledger) -> usize {
    let mut recovered = 0;
    for event in log.entries() {
        if event.sequence < position {
            continue;
        }
        match &event.kind {
            EventKind::Effect { idempotency_key: Some(key), result, .. } => {
                if !ledger.contains(key) {
                    recovered += 1;
                }
                ledger.record(key.clone(), result.clone());
            }
            EventKind::EffectReplayed { idempotency_key, result, .. } => {
                ledger.record(idempotency_key.clone(), result.clone());
            }
            _ => {}
        }
    }
    recovered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::InMemoryEventLog;
    use crate::state::{Checkpoint, InMemoryCheckpointStore};
    use crate::value::Value;

    fn effect(key: &str) -> EventKind {
        EventKind::Effect {
            target: "pay".into(),
            effect: "#irreversible<ledger>".into(),
            idempotency_key: Some(key.into()),
            result: Value::Str(format!("receipt-{key}")),
        }
    }

    #[test]
    fn with_no_checkpoint_the_log_alone_rebuilds_the_ledger() {
        let mut log = InMemoryEventLog::new();
        log.append(None, effect("a"));
        log.append(None, effect("b"));
        let checkpoints = InMemoryCheckpointStore::new();

        let resumption = RecoveryManager::new().resume(&log, &checkpoints);
        assert_eq!(resumption.from_checkpoint, None);
        assert_eq!(resumption.state.ledger.len(), 2);
        assert_eq!(resumption.recovered_from_log, 2);
    }

    #[test]
    fn effects_after_the_checkpoint_are_folded_back_in() {
        // The crash case of §8.3: the write happened, the acknowledgement was
        // lost, and only the log knows.
        let mut log = InMemoryEventLog::new();
        log.append(None, effect("a"));
        let mut state = ExecutionState::new();
        state.ledger.record("a", Value::Str("receipt-a".into()));

        let mut checkpoints = InMemoryCheckpointStore::new();
        checkpoints.save(Checkpoint {
            sequence: 1,
            log_position: log.len() as u64,
            version: 0,
            state,
        });

        log.append(None, effect("b"));

        let resumption = RecoveryManager::new().resume(&log, &checkpoints);
        assert_eq!(resumption.from_checkpoint, Some(1));
        assert!(resumption.state.ledger.contains("a"), "from the checkpoint");
        assert!(resumption.state.ledger.contains("b"), "from the log tail");
        assert_eq!(resumption.recovered_from_log, 1);
    }

    #[test]
    fn a_replayed_effect_stays_in_the_ledger() {
        let mut log = InMemoryEventLog::new();
        log.append(
            None,
            EventKind::EffectReplayed {
                target: "pay".into(),
                idempotency_key: "a".into(),
                result: Value::Int(1),
            },
        );
        let ledger = RecoveryManager::new().ledger_from_log(&log);
        assert!(ledger.contains("a"));
    }

    #[test]
    fn a_resume_leaves_a_mark_in_the_log() {
        let mut log = InMemoryEventLog::new();
        let manager = RecoveryManager::new();
        let resumption = Resumption {
            state: ExecutionState::new(),
            from_checkpoint: Some(4),
            recovered_from_log: 0,
        };
        manager.note_resumption(&mut log, &resumption);
        assert!(matches!(
            log.entries()[0].kind,
            EventKind::Resumed { sequence: 4 }
        ));
    }
}
