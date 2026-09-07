//! Static operation attributes.
//!
//! Attributes are the compile-time half of an operation: everything the
//! compiler can read without running anything. Dynamic data travels as
//! [`Value`](crate::Value) operands instead.

use crate::effect::Effect;
use std::collections::BTreeMap;
use std::fmt;

/// A compile-time constant attached to an operation.
///
/// `BTreeMap` rather than `HashMap` so that printing a module is deterministic:
/// round-trip idempotence is a tested property, not a nicety.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Attribute {
    /// An integer literal.
    Int(i64),
    /// A floating point literal.
    Float(f64),
    /// A boolean literal.
    Bool(bool),
    /// A string literal.
    Str(String),
    /// An effect literal, written `#pure`, `#read_external<web>`, ...
    Effect(Effect),
    /// An ordered list of attributes.
    Array(Vec<Attribute>),
    /// A nested dictionary, keyed in name order.
    Dict(BTreeMap<String, Attribute>),
}

impl Attribute {
    /// The integer payload, if this is an `Int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attribute::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// The floating point payload, widening an `Int` when asked for a number.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Attribute::Float(v) => Some(*v),
            Attribute::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// The boolean payload, if this is a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Attribute::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// The string payload, if this is a `Str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attribute::Str(v) => Some(v),
            _ => None,
        }
    }

    /// The entries, if this is a `Dict`.
    pub fn as_dict(&self) -> Option<&BTreeMap<String, Attribute>> {
        match self {
            Attribute::Dict(d) => Some(d),
            _ => None,
        }
    }

    /// The elements, if this is an `Array`.
    pub fn as_array(&self) -> Option<&[Attribute]> {
        match self {
            Attribute::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl From<i64> for Attribute {
    fn from(v: i64) -> Self {
        Attribute::Int(v)
    }
}

impl From<f64> for Attribute {
    fn from(v: f64) -> Self {
        Attribute::Float(v)
    }
}

impl From<bool> for Attribute {
    fn from(v: bool) -> Self {
        Attribute::Bool(v)
    }
}

impl From<&str> for Attribute {
    fn from(v: &str) -> Self {
        Attribute::Str(v.to_string())
    }
}

impl From<String> for Attribute {
    fn from(v: String) -> Self {
        Attribute::Str(v)
    }
}

impl From<Effect> for Attribute {
    fn from(v: Effect) -> Self {
        Attribute::Effect(v)
    }
}

/// Formats a float so that parsing the output yields the same value, and so
/// that whole numbers keep a decimal point and stay distinguishable from ints.
pub(crate) fn write_float(f: &mut fmt::Formatter<'_>, value: f64) -> fmt::Result {
    if value.is_finite() && value == value.trunc() && value.abs() < 1e15 {
        write!(f, "{value:.1}")
    } else {
        write!(f, "{value}")
    }
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attribute::Int(v) => write!(f, "{v}"),
            Attribute::Float(v) => write_float(f, *v),
            Attribute::Bool(v) => write!(f, "{v}"),
            Attribute::Str(v) => write!(f, "\"{}\"", escape(v)),
            Attribute::Effect(e) => write!(f, "{e}"),
            Attribute::Array(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
            Attribute::Dict(entries) => {
                f.write_str("{")?;
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{key} = {value}")?;
                }
                f.write_str("}")
            }
        }
    }
}

/// Escapes a string for the textual syntax.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

/// An ordered attribute dictionary attached to an operation.
pub type Attributes = BTreeMap<String, Attribute>;
