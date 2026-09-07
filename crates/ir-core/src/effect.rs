//! The effect system of §2.2 — the part that makes optimization decidable.
//!
//! Every operation declares what it does to the world, independently of what it
//! returns. A pass then asks the effect, not a heuristic, whether it is allowed
//! to fire. The rule table of §2.2 is implemented here as [`Effect`] predicates
//! so that no pass has to restate it.
//!
//! ## One documented refinement of the specification
//!
//! §2.2 writes `Irreversible` without a scope. Carrying a scope lets invariant
//! I3 reason about irreversible writes the same way it reasons about
//! `WriteExternal` ones; [`Scope::any`] recovers the unscoped meaning, since an
//! unnamed scope is assumed to overlap every other scope.

use std::fmt;

/// The resource an external effect touches.
///
/// An unnamed scope means "unknown", and an unknown scope conservatively
/// overlaps every other scope — the effect system never guesses in the
/// direction that would let an unsafe pass fire.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Scope(Option<String>);

impl Scope {
    /// A named resource scope.
    pub fn named(name: impl Into<String>) -> Self {
        Scope(Some(name.into()))
    }

    /// The unknown scope: overlaps everything.
    pub fn any() -> Self {
        Scope(None)
    }

    /// The resource name, if this scope is named.
    pub fn name(&self) -> Option<&str> {
        self.0.as_deref()
    }

    /// Whether the scope is the unknown one.
    pub fn is_any(&self) -> bool {
        self.0.is_none()
    }

    /// Whether two scopes may denote the same resource.
    ///
    /// Unknown overlaps everything, which is what keeps `Parallelization` and
    /// `Result Reuse` sound in the presence of an under-annotated program.
    pub fn overlaps(&self, other: &Scope) -> bool {
        match (&self.0, &other.0) {
            (None, _) | (_, None) => true,
            (Some(a), Some(b)) => a == b,
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(name) => f.write_str(name),
            None => f.write_str("*"),
        }
    }
}

/// The effect signature of an operation. See §2.2.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Effect {
    /// No interaction with the outside world: replayable and cacheable.
    Pure,
    /// An external read. Idempotent by nature, but its result can go stale.
    ReadExternal(Scope),
    /// An external write. Not idempotent unless proven otherwise.
    WriteExternal(Scope),
    /// A non-undoable effect: a payment, a deletion, a message sent.
    Irreversible(Scope),
    /// A non-deterministic output: an LLM call, a sampling step.
    Stochastic,
}

/// The effect variant without its payload, for capability grants and reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum EffectClass {
    /// See [`Effect::Pure`].
    Pure,
    /// See [`Effect::ReadExternal`].
    ReadExternal,
    /// See [`Effect::WriteExternal`].
    WriteExternal,
    /// See [`Effect::Irreversible`].
    Irreversible,
    /// See [`Effect::Stochastic`].
    Stochastic,
}

impl fmt::Display for EffectClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EffectClass::Pure => "pure",
            EffectClass::ReadExternal => "read_external",
            EffectClass::WriteExternal => "write_external",
            EffectClass::Irreversible => "irreversible",
            EffectClass::Stochastic => "stochastic",
        })
    }
}

impl std::str::FromStr for EffectClass {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "pure" => EffectClass::Pure,
            "read_external" => EffectClass::ReadExternal,
            "write_external" => EffectClass::WriteExternal,
            "irreversible" => EffectClass::Irreversible,
            "stochastic" => EffectClass::Stochastic,
            _ => return Err(()),
        })
    }
}

impl Effect {
    /// Shorthand for an external read on a named scope.
    pub fn read(scope: impl Into<String>) -> Self {
        Effect::ReadExternal(Scope::named(scope))
    }

    /// Shorthand for an external write on a named scope.
    pub fn write(scope: impl Into<String>) -> Self {
        Effect::WriteExternal(Scope::named(scope))
    }

    /// Shorthand for an irreversible effect on a named scope.
    pub fn irreversible(scope: impl Into<String>) -> Self {
        Effect::Irreversible(Scope::named(scope))
    }

    /// The variant, without its scope.
    pub fn class(&self) -> EffectClass {
        match self {
            Effect::Pure => EffectClass::Pure,
            Effect::ReadExternal(_) => EffectClass::ReadExternal,
            Effect::WriteExternal(_) => EffectClass::WriteExternal,
            Effect::Irreversible(_) => EffectClass::Irreversible,
            Effect::Stochastic => EffectClass::Stochastic,
        }
    }

    /// The scope touched, for the effects that touch one.
    pub fn scope(&self) -> Option<&Scope> {
        match self {
            Effect::ReadExternal(s) | Effect::WriteExternal(s) | Effect::Irreversible(s) => Some(s),
            Effect::Pure | Effect::Stochastic => None,
        }
    }

    /// Whether the effect is `Pure`.
    pub fn is_pure(&self) -> bool {
        matches!(self, Effect::Pure)
    }

    /// Whether the effect can never be undone (§2.2, §7).
    pub fn is_irreversible(&self) -> bool {
        matches!(self, Effect::Irreversible(_))
    }

    /// Whether re-running the operation is observationally free.
    ///
    /// This is the precondition shared by *Dead Action Elimination*,
    /// *Result Reuse / Caching* and *Tool Call Deduplication* in the §5 table.
    pub fn is_replayable(&self) -> bool {
        matches!(self, Effect::Pure | Effect::ReadExternal(_))
    }

    /// Whether a result may be cached, subject to a declared validity window.
    ///
    /// Same condition as [`Effect::is_replayable`], named for the pass that
    /// reads it so the call sites document themselves.
    pub fn is_cacheable(&self) -> bool {
        self.is_replayable()
    }

    /// Whether the operation modifies the outside world.
    pub fn is_write(&self) -> bool {
        matches!(self, Effect::WriteExternal(_) | Effect::Irreversible(_))
    }

    /// Whether the operation may be executed before its guard is known to hold.
    ///
    /// Never for `Irreversible`; a `WriteExternal` needs a compensating action,
    /// which the caller has to supply, so this returns `false` for writes too
    /// and the *Speculative Execution* pass must consult the compensation
    /// attribute explicitly.
    pub fn is_speculatable(&self) -> bool {
        self.is_replayable()
    }

    /// Whether the operation requires a [`Capability`](crate::Capability)
    /// before it may be lowered (§2.3, invariant I2).
    pub fn requires_capability(&self) -> bool {
        !matches!(self, Effect::Pure)
    }

    /// Whether two effects conflict when scheduled concurrently.
    ///
    /// This is invariant I3 in one place: two operations may share a
    /// `control.parallel` region unless at least one writes a scope the other
    /// also touches.
    pub fn conflicts_with(&self, other: &Effect) -> bool {
        let (Some(a), Some(b)) = (self.scope(), other.scope()) else {
            return false;
        };
        (self.is_write() || other.is_write()) && a.overlaps(b)
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Pure => f.write_str("#pure"),
            Effect::Stochastic => f.write_str("#stochastic"),
            Effect::ReadExternal(s) | Effect::WriteExternal(s) | Effect::Irreversible(s) => {
                write!(f, "#{}", self.class())?;
                match s.name() {
                    Some(name) => write!(f, "<{name}>"),
                    None => Ok(()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_scope_overlaps_everything() {
        assert!(Scope::any().overlaps(&Scope::named("db")));
        assert!(Scope::named("db").overlaps(&Scope::any()));
        assert!(!Scope::named("db").overlaps(&Scope::named("cache")));
    }

    #[test]
    fn reads_never_conflict_with_reads() {
        let a = Effect::read("db");
        let b = Effect::read("db");
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn a_write_conflicts_with_any_touch_of_the_same_scope() {
        let write = Effect::write("db");
        assert!(write.conflicts_with(&Effect::read("db")));
        assert!(write.conflicts_with(&Effect::write("db")));
        assert!(write.conflicts_with(&Effect::irreversible("db")));
        assert!(!write.conflicts_with(&Effect::read("cache")));
        assert!(!write.conflicts_with(&Effect::Pure));
        assert!(!write.conflicts_with(&Effect::Stochastic));
    }

    #[test]
    fn an_unscoped_write_conflicts_with_every_scoped_effect() {
        let write = Effect::WriteExternal(Scope::any());
        assert!(write.conflicts_with(&Effect::read("db")));
        assert!(write.conflicts_with(&Effect::read("cache")));
    }

    #[test]
    fn only_pure_and_read_are_replayable() {
        assert!(Effect::Pure.is_replayable());
        assert!(Effect::read("web").is_replayable());
        assert!(!Effect::write("db").is_replayable());
        assert!(!Effect::irreversible("db").is_replayable());
        assert!(!Effect::Stochastic.is_replayable());
    }

    #[test]
    fn effects_round_trip_through_display() {
        assert_eq!(Effect::Pure.to_string(), "#pure");
        assert_eq!(Effect::read("web").to_string(), "#read_external<web>");
        assert_eq!(Effect::WriteExternal(Scope::any()).to_string(), "#write_external");
        assert_eq!(Effect::irreversible("db").to_string(), "#irreversible<db>");
        assert_eq!(Effect::Stochastic.to_string(), "#stochastic");
    }
}
