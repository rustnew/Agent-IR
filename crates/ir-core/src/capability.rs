//! Capabilities: the structural security boundary of §2.3 and §7.
//!
//! An operation may only be lowered to the runtime if the agent holds a
//! capability covering its declared effect. The check happens at compile time,
//! before any network call — that is the guarantee the whole design exists to
//! provide.

use crate::effect::{Effect, EffectClass, Scope};
use std::collections::BTreeMap;

/// A permission granted to an agent (§2.3).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Capability {
    /// The name an `agent.verify` refers to.
    pub name: String,
    /// The resource this capability applies to.
    pub scope: Scope,
    /// The effect classes it authorizes.
    pub grants: Vec<EffectClass>,
    /// When set, a human must approve before the operation may run, and the
    /// compiler rejects the program unless approval has been recorded.
    pub requires_approval: bool,
}

impl Capability {
    /// A capability granting one effect class over one named scope.
    pub fn new(name: impl Into<String>, scope: Scope, grants: impl Into<Vec<EffectClass>>) -> Self {
        Capability {
            name: name.into(),
            scope,
            grants: grants.into(),
            requires_approval: false,
        }
    }

    /// The same capability, but gated behind a human decision.
    pub fn requiring_approval(mut self) -> Self {
        self.requires_approval = true;
        self
    }

    /// Whether this capability authorizes the given effect.
    ///
    /// The scope must *contain* the effect's scope, not merely overlap it: an
    /// overlap test would let a capability on an unknown scope be satisfied by
    /// anything, which is the wrong direction for a permission check.
    pub fn covers(&self, effect: &Effect) -> bool {
        if !self.grants.contains(&effect.class()) {
            return false;
        }
        match effect.scope() {
            None => true,
            Some(target) => self.scope.is_any() || self.scope.name() == target.name(),
        }
    }
}

/// The set of capabilities an agent holds, keyed by name.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapabilitySet {
    capabilities: BTreeMap<String, Capability>,
    /// Capability names a human has explicitly approved for this session.
    approvals: Vec<String>,
}

impl CapabilitySet {
    /// An agent holding nothing: every non-`Pure` operation will be rejected.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Adds a capability, replacing any capability of the same name.
    pub fn grant(&mut self, capability: Capability) -> &mut Self {
        self.capabilities
            .insert(capability.name.clone(), capability);
        self
    }

    /// Records a human approval for a capability that requires one.
    pub fn approve(&mut self, name: impl Into<String>) -> &mut Self {
        let name = name.into();
        if !self.approvals.contains(&name) {
            self.approvals.push(name);
        }
        self
    }

    /// Looks a capability up by name.
    pub fn get(&self, name: &str) -> Option<&Capability> {
        self.capabilities.get(name)
    }

    /// Whether a human has approved the named capability.
    pub fn is_approved(&self, name: &str) -> bool {
        self.approvals.iter().any(|n| n == name)
    }

    /// Every capability held, in name order.
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.values()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }

    /// The capability that authorizes `effect`, if the agent holds one.
    pub fn covering(&self, effect: &Effect) -> Option<&Capability> {
        self.capabilities.values().find(|c| c.covers(effect))
    }

    /// Whether the named capability exists, covers the effect, and — if it is
    /// gated — has been approved.
    pub fn authorizes(&self, name: &str, effect: &Effect) -> bool {
        match self.get(name) {
            Some(cap) => cap.covers(effect) && (!cap.requires_approval || self.is_approved(name)),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delete_db() -> Capability {
        Capability::new(
            "delete_database",
            Scope::named("db"),
            [EffectClass::Irreversible],
        )
    }

    #[test]
    fn a_capability_covers_only_the_class_it_grants() {
        let cap = delete_db();
        assert!(cap.covers(&Effect::irreversible("db")));
        assert!(!cap.covers(&Effect::write("db")));
    }

    #[test]
    fn a_capability_does_not_cover_another_scope() {
        let cap = delete_db();
        assert!(!cap.covers(&Effect::irreversible("cache")));
    }

    #[test]
    fn an_unknown_effect_scope_is_not_satisfied_by_a_narrow_capability() {
        let cap = delete_db();
        assert!(!cap.covers(&Effect::Irreversible(Scope::any())));
    }

    #[test]
    fn approval_gates_authorization() {
        let mut set = CapabilitySet::empty();
        set.grant(delete_db().requiring_approval());
        let effect = Effect::irreversible("db");
        assert!(!set.authorizes("delete_database", &effect));
        set.approve("delete_database");
        assert!(set.authorizes("delete_database", &effect));
    }

    #[test]
    fn an_empty_set_authorizes_nothing() {
        let set = CapabilitySet::empty();
        assert!(set.covering(&Effect::irreversible("db")).is_none());
    }
}
