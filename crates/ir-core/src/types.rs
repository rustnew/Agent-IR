//! The value type lattice of §2.1.
//!
//! The set is deliberately closed: a program may not invent a type, because the
//! verifier has to be able to decide, for every value, whether it is allowed to
//! cross an effect boundary. [`Type::Unknown`] is the single escape hatch, and
//! it exists precisely because an LLM-proposed plan contains values whose type
//! is only knowable after verification.

use std::fmt;

/// Element type of a [`Type::Tensor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// Boolean element, printed `i1`.
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::Bool => "i1",
        })
    }
}

/// A value type. See §2.1 of the specification.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Type {
    /// Scalar integer.
    Int,
    /// Scalar float.
    Float,
    /// Scalar boolean.
    Bool,
    /// Scalar string.
    Str,
    /// A dense tensor.
    Tensor {
        /// Extent of each dimension.
        shape: Vec<i64>,
        /// Element type.
        dtype: DType,
    },
    /// A handle on an external resource (a model, a database, a candidate...).
    Ref {
        /// The name of the external resource being referred to.
        resource: String,
    },
    /// An observation carrying a named schema.
    Observation {
        /// The name of the schema the observation conforms to.
        schema: String,
    },
    /// A plan proposed by the LLM builder.
    Plan,
    /// The result of a tool call, carrying a named schema.
    ToolResult {
        /// The name of the schema the result conforms to.
        schema: String,
    },
    /// A handle into the memory store.
    Memory,
    /// Produced by the LLM, not yet typed. The verifier must resolve every
    /// `Unknown` before the value feeds a non-`Pure` operation (invariant I1).
    Unknown,
}

impl Type {
    /// A tensor type with the given shape and element type.
    pub fn tensor(shape: impl Into<Vec<i64>>, dtype: DType) -> Self {
        Type::Tensor {
            shape: shape.into(),
            dtype,
        }
    }

    /// A reference to an external resource.
    pub fn reference(resource: impl Into<String>) -> Self {
        Type::Ref {
            resource: resource.into(),
        }
    }

    /// An observation with the given schema name.
    pub fn observation(schema: impl Into<String>) -> Self {
        Type::Observation {
            schema: schema.into(),
        }
    }

    /// A tool result with the given schema name.
    pub fn tool_result(schema: impl Into<String>) -> Self {
        Type::ToolResult {
            schema: schema.into(),
        }
    }

    /// Whether the type is still to be resolved by the verifier.
    pub fn is_unknown(&self) -> bool {
        matches!(self, Type::Unknown)
    }

    /// Whether the type is one of the scalars of §2.1.
    pub fn is_scalar(&self) -> bool {
        matches!(self, Type::Int | Type::Float | Type::Bool | Type::Str)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => f.write_str("!core.int"),
            Type::Float => f.write_str("!core.float"),
            Type::Bool => f.write_str("!core.bool"),
            Type::Str => f.write_str("!core.string"),
            Type::Tensor { shape, dtype } => {
                // Comma separated rather than MLIR's `2x3xf32`: a dimension
                // list that lexes as ordinary tokens keeps the parser regular.
                f.write_str("!core.tensor<")?;
                for dim in shape {
                    write!(f, "{dim}, ")?;
                }
                write!(f, "{dtype}>")
            }
            Type::Ref { resource } => write!(f, "!tool.ref<{resource}>"),
            Type::Observation { schema } => write!(f, "!observation.observation<{schema}>"),
            Type::Plan => f.write_str("!agent.plan"),
            Type::ToolResult { schema } => write!(f, "!tool.result<{schema}>"),
            Type::Memory => f.write_str("!memory.memory"),
            Type::Unknown => f.write_str("!core.unknown"),
        }
    }
}
