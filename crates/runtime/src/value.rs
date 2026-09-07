//! Runtime values.
//!
//! The IR types of §2.1 describe what a value *is*; this is what one *holds*
//! while the program runs. The two are deliberately separate: the compiler
//! reasons about `!tool.result<benchmark>`, the runtime carries the record that
//! came back.

use agent_ir_core::Attribute;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// A value produced or consumed while the program runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Value {
    /// Absent, or an operation that produced nothing meaningful.
    Null,
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A string.
    Str(String),
    /// An ordered list, e.g. the candidates a `control.loop` iterates.
    List(Vec<Value>),
    /// A record, e.g. a tool result with named fields.
    Record(BTreeMap<String, Value>),
}

impl Value {
    /// A record from pairs.
    pub fn record<K: Into<String>>(entries: impl IntoIterator<Item = (K, Value)>) -> Self {
        Value::Record(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// A list from anything iterable.
    pub fn list(items: impl IntoIterator<Item = Value>) -> Self {
        Value::List(items.into_iter().collect())
    }

    /// The integer payload, if there is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// The numeric payload, widening an integer.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// The boolean payload, if there is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// The string payload, if there is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(v) => Some(v),
            _ => None,
        }
    }

    /// The elements, if this is a list.
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(items) => Some(items),
            _ => None,
        }
    }

    /// One named field of a record.
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Record(entries) => entries.get(name),
            _ => None,
        }
    }

    /// Whether the value counts as true for `control.if`.
    ///
    /// Only a genuine boolean does. A truthiness rule that accepted numbers or
    /// non-empty strings would make a mis-typed condition silently take a
    /// branch, which is exactly the class of bug the type system exists to
    /// surface.
    pub fn is_true(&self) -> Option<bool> {
        self.as_bool()
    }

    /// A stable rendering, used as the identity half of a loop guard key (§8.4).
    pub fn fingerprint(&self) -> String {
        match self {
            Value::Null => "null".into(),
            Value::Int(v) => format!("i{v}"),
            Value::Float(v) => format!("f{v}"),
            Value::Bool(v) => format!("b{v}"),
            Value::Str(v) => format!("s{v}"),
            Value::List(items) => {
                let inner: Vec<String> = items.iter().map(Value::fingerprint).collect();
                format!("[{}]", inner.join(","))
            }
            Value::Record(entries) => {
                let inner: Vec<String> = entries
                    .iter()
                    .map(|(k, v)| format!("{k}={}", v.fingerprint()))
                    .collect();
                format!("{{{}}}", inner.join(","))
            }
        }
    }
}

impl From<&Attribute> for Value {
    fn from(attribute: &Attribute) -> Self {
        match attribute {
            Attribute::Int(v) => Value::Int(*v),
            Attribute::Float(v) => Value::Float(*v),
            Attribute::Bool(v) => Value::Bool(*v),
            Attribute::Str(v) => Value::Str(v.clone()),
            Attribute::Effect(e) => Value::Str(e.to_string()),
            Attribute::Array(items) => Value::List(items.iter().map(Value::from).collect()),
            Attribute::Dict(entries) => Value::Record(
                entries
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(v)))
                    .collect(),
            ),
        }
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => f.write_str("null"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Str(v) => write!(f, "\"{v}\""),
            Value::List(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
            Value::Record(entries) => {
                f.write_str("{")?;
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{key}: {value}")?;
                }
                f.write_str("}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_boolean_is_a_condition() {
        assert_eq!(Value::Bool(true).is_true(), Some(true));
        assert_eq!(Value::Int(1).is_true(), None);
        assert_eq!(Value::Str("yes".into()).is_true(), None);
    }

    #[test]
    fn fingerprints_are_stable_and_distinguishing() {
        let a = Value::record([("x", Value::Int(1)), ("y", Value::Int(2))]);
        let b = Value::record([("y", Value::Int(2)), ("x", Value::Int(1))]);
        assert_eq!(a.fingerprint(), b.fingerprint(), "field order must not matter");
        assert_ne!(a.fingerprint(), Value::record([("x", Value::Int(2))]).fingerprint());
    }

    #[test]
    fn attributes_convert_into_values() {
        assert_eq!(Value::from(&Attribute::Int(3)), Value::Int(3));
        assert_eq!(
            Value::from(&Attribute::Array(vec![Attribute::Bool(true)])),
            Value::List(vec![Value::Bool(true)])
        );
    }

    #[test]
    fn a_value_round_trips_through_json() {
        let value = Value::record([
            ("latency", Value::Float(1.5)),
            ("tags", Value::list([Value::Str("a".into())])),
        ]);
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&json).unwrap(), value);
    }
}
