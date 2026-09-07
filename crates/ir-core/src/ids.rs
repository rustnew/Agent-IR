//! Dense arena handles for the IR entities.
//!
//! Every entity lives in a `Vec` owned by the [`Module`](crate::Module); an id is
//! an index into that vector. Ids are cheap to copy, cheap to compare and stable
//! for the lifetime of the module, which is what lets analyses build dense side
//! tables instead of hashing pointers.

use std::fmt;

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
        pub struct $name(pub(crate) u32);

        impl $name {
            /// Wraps a raw index. Only the owning module should mint ids.
            pub(crate) fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).expect("Agent IR module exceeded 2^32 entities"))
            }

            /// The raw index this id refers to.
            #[inline]
            pub fn index(self) -> usize {
                self.0 as usize
            }

            /// The numeric form used in diagnostics and the textual syntax.
            #[inline]
            pub fn number(self) -> u32 {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
}

define_id!(
    /// Identifies an [`Operation`](crate::Operation) within a module.
    OperationId, "op"
);
define_id!(
    /// Identifies a [`Value`](crate::Value) within a module.
    ValueId, "v"
);
define_id!(
    /// Identifies a [`Block`](crate::Block) within a module.
    BlockId, "bb"
);
define_id!(
    /// Identifies a [`Region`](crate::Region) within a module.
    RegionId, "r"
);
