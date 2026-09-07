//! The `Module → Region → Block → Operation → Value` hierarchy of §3.1.
//!
//! Everything lives in arenas owned by the [`Module`]; entities refer to each
//! other by index. Erasing an operation leaves a tombstone rather than shifting
//! the arena, so ids stay valid for the whole compilation and analyses can use
//! dense side tables.

use crate::attribute::{Attribute, Attributes};
use crate::capability::CapabilitySet;
use crate::effect::Effect;
use crate::ids::{BlockId, OperationId, RegionId, ValueId};
use crate::provenance::Provenance;
use crate::types::Type;
use std::fmt;

/// A fully qualified operation name, `dialect.name`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct OpName {
    /// The dialect half, e.g. `agent`.
    pub dialect: String,
    /// The operation half, e.g. `action`.
    pub name: String,
}

impl OpName {
    /// Builds a name from its two halves.
    pub fn new(dialect: impl Into<String>, name: impl Into<String>) -> Self {
        OpName { dialect: dialect.into(), name: name.into() }
    }

    /// Whether this is the given `dialect.name`.
    pub fn is(&self, dialect: &str, name: &str) -> bool {
        self.dialect == dialect && self.name == name
    }

    /// Whether the operation belongs to the given dialect.
    pub fn in_dialect(&self, dialect: &str) -> bool {
        self.dialect == dialect
    }
}

impl fmt::Display for OpName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.dialect, self.name)
    }
}

impl std::str::FromStr for OpName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once('.') {
            Some((dialect, name)) if !dialect.is_empty() && !name.is_empty() => {
                Ok(OpName::new(dialect, name))
            }
            _ => Err(format!("`{s}` is not a `dialect.name` operation name")),
        }
    }
}

/// What defines a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ValueDef {
    /// The `index`-th result of an operation.
    OpResult {
        /// The producing operation.
        op: OperationId,
        /// Which of its results this is.
        index: usize,
    },
    /// The `index`-th argument of a block.
    BlockArg {
        /// The owning block.
        block: BlockId,
        /// Which of its arguments this is.
        index: usize,
    },
}

/// A single-assignment value (§3.1).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Value {
    /// This value's handle.
    pub id: ValueId,
    /// The name used in the textual syntax, without the leading `%`.
    pub name: Option<String>,
    /// The value's type (§2.1).
    pub ty: Type,
    /// Where the value came from (§2.4).
    pub provenance: Provenance,
    /// What defines it.
    pub def: ValueDef,
}

impl Value {
    /// The operation that produced this value, if it is an operation result.
    pub fn producer(&self) -> Option<OperationId> {
        match self.def {
            ValueDef::OpResult { op, .. } => Some(op),
            ValueDef::BlockArg { .. } => None,
        }
    }

    /// The block this value is an argument of, if it is a block argument.
    pub fn owning_block(&self) -> Option<BlockId> {
        match self.def {
            ValueDef::BlockArg { block, .. } => Some(block),
            ValueDef::OpResult { .. } => None,
        }
    }
}

/// An operation (§3.1).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Operation {
    /// This operation's handle.
    pub id: OperationId,
    /// The fully qualified `dialect.name`.
    pub name: OpName,
    /// A literal that names what the operation does, e.g. the tool being
    /// called. Printed as `agent.action "inspect_model"(...)`.
    pub literal: Option<String>,
    /// The values consumed, in order.
    pub operands: Vec<ValueId>,
    /// The values produced, in order.
    pub results: Vec<ValueId>,
    /// Compile-time attributes, in name order.
    pub attributes: Attributes,
    /// The declared effect signature (§2.2).
    pub effect: Effect,
    /// Nested regions, for `control.if`, `control.loop` and `control.parallel`.
    pub regions: Vec<RegionId>,
    /// The block this operation sits in; `None` for an erased operation or for
    /// one not yet inserted.
    pub parent: Option<BlockId>,
    /// Erased operations keep their arena slot so ids stay stable.
    pub erased: bool,
}

impl Operation {
    /// Reads an attribute by name.
    pub fn attr(&self, key: &str) -> Option<&Attribute> {
        self.attributes.get(key)
    }

    /// Reads a string attribute.
    pub fn str_attr(&self, key: &str) -> Option<&str> {
        self.attr(key).and_then(Attribute::as_str)
    }

    /// Reads an integer attribute.
    pub fn int_attr(&self, key: &str) -> Option<i64> {
        self.attr(key).and_then(Attribute::as_int)
    }

    /// Reads a numeric attribute, widening integers.
    pub fn float_attr(&self, key: &str) -> Option<f64> {
        self.attr(key).and_then(Attribute::as_float)
    }

    /// Reads a boolean attribute.
    pub fn bool_attr(&self, key: &str) -> Option<bool> {
        self.attr(key).and_then(Attribute::as_bool)
    }

    /// Sets an attribute, replacing any previous value.
    pub fn set_attr(&mut self, key: impl Into<String>, value: impl Into<Attribute>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// The identity used by *Tool Call Deduplication* and *Result Reuse*: the
    /// operation name, its literal, its operands and its effect scope (§5).
    ///
    /// Two operations sharing a key compute the same thing, provided their
    /// effect is replayable — which the pass, not this function, must check.
    pub fn cache_key(&self) -> CacheKey {
        CacheKey {
            name: self.name.clone(),
            literal: self.literal.clone(),
            operands: self.operands.clone(),
            scope: self.effect.scope().map(ToString::to_string),
            attributes: self
                .attributes
                .iter()
                .filter(|(k, _)| k.as_str() != "effect")
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect(),
        }
    }
}

/// The identity of a computation, for caching and deduplication (§5).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// The operation name.
    pub name: OpName,
    /// The operation literal, e.g. the tool being called.
    pub literal: Option<String>,
    /// The operands, by identity.
    pub operands: Vec<ValueId>,
    /// The effect scope, if the effect has one.
    pub scope: Option<String>,
    /// Every attribute except `effect`, rendered for comparison.
    pub attributes: Vec<(String, String)>,
}

/// A block: an ordered list of operations, plus its arguments (§3.1).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Block {
    /// This block's handle.
    pub id: BlockId,
    /// The block arguments, in order.
    pub args: Vec<ValueId>,
    /// The operations it holds, in program order.
    pub ops: Vec<OperationId>,
    /// The region this block belongs to.
    pub parent: RegionId,
}

/// A region: an ordered list of blocks owned by an operation (§3.1).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Region {
    /// This region's handle.
    pub id: RegionId,
    /// The blocks it holds, in order.
    pub blocks: Vec<BlockId>,
    /// The operation owning this region; `None` for the module body.
    pub parent: Option<OperationId>,
}

/// A complete IR module (§3.1).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Module {
    /// The module symbol name.
    pub name: String,
    /// The IR version: `IR₀`, `IR₁`, ... of §1.2. Bumped by the runtime each
    /// time an observation produces a new program.
    pub version: u64,
    /// The capabilities the agent holds while this module executes (§2.3).
    pub capabilities: CapabilitySet,
    ops: Vec<Operation>,
    values: Vec<Value>,
    blocks: Vec<Block>,
    regions: Vec<Region>,
    body: RegionId,
}

impl Module {
    /// An empty module with a single-block body region.
    pub fn new(name: impl Into<String>) -> Self {
        let mut module = Module {
            name: name.into(),
            version: 0,
            capabilities: CapabilitySet::empty(),
            ops: Vec::new(),
            values: Vec::new(),
            blocks: Vec::new(),
            regions: Vec::new(),
            body: RegionId(0),
        };
        let body = module.create_region(None);
        module.create_block(body);
        module.body = body;
        module
    }

    /// The module body region, holding the top-level operations.
    pub fn body(&self) -> RegionId {
        self.body
    }

    /// The single block of the module body.
    pub fn body_block(&self) -> BlockId {
        self.region(self.body).blocks[0]
    }

    // ---------------------------------------------------------------- reads

    /// An operation by id.
    pub fn op(&self, id: OperationId) -> &Operation {
        &self.ops[id.index()]
    }

    /// An operation by id, mutably.
    pub fn op_mut(&mut self, id: OperationId) -> &mut Operation {
        &mut self.ops[id.index()]
    }

    /// A value by id.
    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.index()]
    }

    /// A value by id, mutably.
    pub fn value_mut(&mut self, id: ValueId) -> &mut Value {
        &mut self.values[id.index()]
    }

    /// A block by id.
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.index()]
    }

    /// A block by id, mutably.
    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.index()]
    }

    /// A region by id.
    pub fn region(&self, id: RegionId) -> &Region {
        &self.regions[id.index()]
    }

    /// A region by id, mutably.
    pub fn region_mut(&mut self, id: RegionId) -> &mut Region {
        &mut self.regions[id.index()]
    }

    /// Every operation ever created, erased ones included. Prefer
    /// [`Module::walk`] to visit the live program.
    pub fn all_ops(&self) -> impl Iterator<Item = &Operation> {
        self.ops.iter()
    }

    /// Every value ever created.
    pub fn all_values(&self) -> impl Iterator<Item = &Value> {
        self.values.iter()
    }

    /// How many operation slots exist, erased ones included. Analyses size
    /// their side tables with this.
    pub fn op_capacity(&self) -> usize {
        self.ops.len()
    }

    /// How many value slots exist.
    pub fn value_capacity(&self) -> usize {
        self.values.len()
    }

    /// How many regions the module holds.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// A region by arena position, for traversals that need every region.
    pub fn region_by_index(&self, index: usize) -> &Region {
        &self.regions[index]
    }

    /// How many blocks the module holds.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    // ------------------------------------------------------------- creation

    /// Creates a region, optionally owned by an operation.
    pub fn create_region(&mut self, parent: Option<OperationId>) -> RegionId {
        let id = RegionId::from_index(self.regions.len());
        self.regions.push(Region { id, blocks: Vec::new(), parent });
        id
    }

    /// Creates a block at the end of a region.
    pub fn create_block(&mut self, region: RegionId) -> BlockId {
        let id = BlockId::from_index(self.blocks.len());
        self.blocks.push(Block { id, args: Vec::new(), ops: Vec::new(), parent: region });
        self.region_mut(region).blocks.push(id);
        id
    }

    /// Appends an argument to a block and returns the value that names it.
    pub fn add_block_arg(
        &mut self,
        block: BlockId,
        name: Option<String>,
        ty: Type,
        provenance: Provenance,
    ) -> ValueId {
        let index = self.block(block).args.len();
        let id = ValueId::from_index(self.values.len());
        self.values.push(Value {
            id,
            name,
            ty,
            provenance,
            def: ValueDef::BlockArg { block, index },
        });
        self.block_mut(block).args.push(id);
        id
    }

    /// Creates an operation without inserting it into a block.
    ///
    /// Callers normally go through [`crate::Builder`], which inserts as it
    /// builds; this is the primitive the builder and the parser share.
    pub fn create_op(
        &mut self,
        name: OpName,
        literal: Option<String>,
        operands: Vec<ValueId>,
        result_types: Vec<(Option<String>, Type, Provenance)>,
        attributes: Attributes,
        effect: Effect,
    ) -> OperationId {
        let id = OperationId::from_index(self.ops.len());
        let mut results = Vec::with_capacity(result_types.len());
        for (index, (value_name, ty, mut provenance)) in result_types.into_iter().enumerate() {
            let value_id = ValueId::from_index(self.values.len());
            provenance.producer = Some(id);
            self.values.push(Value {
                id: value_id,
                name: value_name,
                ty,
                provenance,
                def: ValueDef::OpResult { op: id, index },
            });
            results.push(value_id);
        }
        self.ops.push(Operation {
            id,
            name,
            literal,
            operands,
            results,
            attributes,
            effect,
            regions: Vec::new(),
            parent: None,
            erased: false,
        });
        id
    }

    /// Creates a region owned by `op` and records it on the operation.
    pub fn create_op_region(&mut self, op: OperationId) -> RegionId {
        let region = self.create_region(Some(op));
        self.op_mut(op).regions.push(region);
        region
    }

    // ------------------------------------------------------------- mutation

    /// Appends an operation to the end of a block.
    pub fn append_to_block(&mut self, block: BlockId, op: OperationId) {
        debug_assert!(self.op(op).parent.is_none(), "operation is already in a block");
        self.op_mut(op).parent = Some(block);
        self.block_mut(block).ops.push(op);
    }

    /// Inserts an operation at a position in a block.
    pub fn insert_in_block(&mut self, block: BlockId, index: usize, op: OperationId) {
        debug_assert!(self.op(op).parent.is_none(), "operation is already in a block");
        self.op_mut(op).parent = Some(block);
        self.block_mut(block).ops.insert(index, op);
    }

    /// Detaches an operation from its block without erasing it.
    pub fn detach(&mut self, op: OperationId) {
        if let Some(block) = self.op(op).parent {
            self.block_mut(block).ops.retain(|&candidate| candidate != op);
            self.op_mut(op).parent = None;
        }
    }

    /// Removes an operation from the program.
    ///
    /// The arena slot survives as a tombstone so that ids already handed out
    /// stay meaningful in diagnostics and pass reports.
    pub fn erase(&mut self, op: OperationId) {
        self.detach(op);
        self.op_mut(op).erased = true;
    }

    /// Moves an operation to the end of another block.
    pub fn move_to_end(&mut self, op: OperationId, block: BlockId) {
        self.detach(op);
        self.append_to_block(block, op);
    }

    /// The position of an operation inside its block.
    pub fn position_in_block(&self, op: OperationId) -> Option<usize> {
        let block = self.op(op).parent?;
        self.block(block).ops.iter().position(|&candidate| candidate == op)
    }

    /// Rewrites every use of `from` into a use of `to`.
    ///
    /// Returns how many operand slots changed. Used by *Result Reuse* and
    /// *Tool Call Deduplication* once they have proven two computations equal.
    pub fn replace_all_uses(&mut self, from: ValueId, to: ValueId) -> usize {
        let mut replaced = 0;
        for op in &mut self.ops {
            if op.erased {
                continue;
            }
            for operand in &mut op.operands {
                if *operand == from {
                    *operand = to;
                    replaced += 1;
                }
            }
        }
        replaced
    }

    // --------------------------------------------------------------- walking

    /// Visits every live operation in program order, entering nested regions.
    pub fn walk(&self, mut visit: impl FnMut(&Operation)) {
        self.walk_region(self.body, &mut visit);
    }

    fn walk_region(&self, region: RegionId, visit: &mut impl FnMut(&Operation)) {
        for &block in &self.region(region).blocks {
            for &op_id in &self.block(block).ops {
                let op = self.op(op_id);
                if op.erased {
                    continue;
                }
                visit(op);
                for &nested in &op.regions {
                    self.walk_region(nested, visit);
                }
            }
        }
    }

    /// The ids of every live operation, in program order.
    pub fn op_ids(&self) -> Vec<OperationId> {
        let mut ids = Vec::new();
        self.walk(|op| ids.push(op.id));
        ids
    }

    /// The ids of the live operations directly inside a region, in order.
    pub fn ops_in_region(&self, region: RegionId) -> Vec<OperationId> {
        self.region(region)
            .blocks
            .iter()
            .flat_map(|&block| self.block(block).ops.iter().copied())
            .filter(|&op| !self.op(op).erased)
            .collect()
    }

    /// The operation owning the region a given operation sits in, if any.
    pub fn parent_op(&self, op: OperationId) -> Option<OperationId> {
        let block = self.op(op).parent?;
        self.region(self.block(block).parent).parent
    }

    /// Walks outward from an operation through its enclosing operations.
    pub fn ancestors(&self, op: OperationId) -> Vec<OperationId> {
        let mut chain = Vec::new();
        let mut current = op;
        while let Some(parent) = self.parent_op(current) {
            chain.push(parent);
            current = parent;
        }
        chain
    }

    /// The top-level operations of the module body (the functions).
    pub fn top_level(&self) -> Vec<OperationId> {
        self.ops_in_region(self.body)
    }

    /// Looks up a top-level `agent.func` by name.
    ///
    /// The name is the operation literal — `agent.func "optimize_inference"` —
    /// rather than a separate attribute, so there is only one place for it to
    /// be wrong.
    pub fn function(&self, name: &str) -> Option<OperationId> {
        self.top_level().into_iter().find(|&id| {
            let op = self.op(id);
            op.name.is("agent", "func") && op.literal.as_deref() == Some(name)
        })
    }
}
