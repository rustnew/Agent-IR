//! A fluent builder for constructing modules in Rust.
//!
//! The parser and the LLM-facing front end both end up calling the same
//! primitives on [`Module`]; this builder is the ergonomic face of them, and is
//! what tests and the SDK use to write programs by hand.

use crate::attribute::{Attribute, Attributes};
use crate::effect::Effect;
use crate::ids::{BlockId, OperationId, RegionId, ValueId};
use crate::module::{Module, OpName};
use crate::provenance::Provenance;
use crate::types::Type;

/// Builds operations into a module at a chosen insertion point.
pub struct Builder<'m> {
    module: &'m mut Module,
    insertion: BlockId,
}

impl<'m> Builder<'m> {
    /// Builds into the module body.
    pub fn new(module: &'m mut Module) -> Self {
        let insertion = module.body_block();
        Builder { module, insertion }
    }

    /// Builds into a specific block.
    pub fn at_block(module: &'m mut Module, block: BlockId) -> Self {
        Builder { module, insertion: block }
    }

    /// The module being built.
    pub fn module(&mut self) -> &mut Module {
        self.module
    }

    /// The block operations are currently appended to.
    pub fn insertion_point(&self) -> BlockId {
        self.insertion
    }

    /// Redirects the builder at another block.
    pub fn set_insertion_point(&mut self, block: BlockId) {
        self.insertion = block;
    }

    /// Starts building an operation named `dialect.name`.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not of the form `dialect.name`. Operation names are
    /// compile-time constants at every call site, so a malformed one is a bug
    /// in the caller rather than a runtime condition.
    pub fn op(&mut self, name: &str) -> OpBuilder<'_, 'm> {
        let name: OpName = name.parse().expect("operation name must be `dialect.name`");
        OpBuilder {
            builder: self,
            name,
            literal: None,
            operands: Vec::new(),
            results: Vec::new(),
            attributes: Attributes::new(),
            effect: Effect::Pure,
        }
    }

    /// Builds an `agent.func` and fills its body.
    ///
    /// The closure receives a builder positioned inside the function entry
    /// block, plus the block arguments in declaration order.
    pub fn func(
        &mut self,
        name: &str,
        params: impl IntoIterator<Item = (&'static str, Type)>,
        body: impl FnOnce(&mut Builder<'_>, &[ValueId]),
    ) -> OperationId {
        let func = self.op("agent.func").literal(name).build();
        let region = self.module.create_op_region(func);
        let entry = self.module.create_block(region);
        let args: Vec<ValueId> = params
            .into_iter()
            .map(|(param, ty)| {
                self.module.add_block_arg(
                    entry,
                    Some(param.to_string()),
                    ty,
                    Provenance::from_user(),
                )
            })
            .collect();
        let mut inner = Builder::at_block(self.module, entry);
        body(&mut inner, &args);
        func
    }

    /// Adds a region to an operation and fills it.
    ///
    /// The closure receives a builder positioned inside the region's entry
    /// block. Used for `control.parallel`, `control.if` and `control.loop`.
    pub fn region(
        &mut self,
        op: OperationId,
        args: impl IntoIterator<Item = (String, Type)>,
        body: impl FnOnce(&mut Builder<'_>, &[ValueId]),
    ) -> RegionId {
        let region = self.module.create_op_region(op);
        let block = self.module.create_block(region);
        let values: Vec<ValueId> = args
            .into_iter()
            .map(|(name, ty)| {
                self.module.add_block_arg(block, Some(name), ty, Provenance::from_tool())
            })
            .collect();
        let mut inner = Builder::at_block(self.module, block);
        body(&mut inner, &values);
        region
    }
}

/// A partially specified operation. Finish with [`OpBuilder::build`] or
/// [`OpBuilder::build_one`].
pub struct OpBuilder<'b, 'm> {
    builder: &'b mut Builder<'m>,
    name: OpName,
    literal: Option<String>,
    operands: Vec<ValueId>,
    results: Vec<(Option<String>, Type, Provenance)>,
    attributes: Attributes,
    effect: Effect,
}

impl OpBuilder<'_, '_> {
    /// Sets the literal, e.g. the tool name in `tool.call "benchmark"`.
    pub fn literal(mut self, literal: impl Into<String>) -> Self {
        self.literal = Some(literal.into());
        self
    }

    /// Appends an operand.
    pub fn operand(mut self, value: ValueId) -> Self {
        self.operands.push(value);
        self
    }

    /// Appends several operands.
    pub fn operands(mut self, values: impl IntoIterator<Item = ValueId>) -> Self {
        self.operands.extend(values);
        self
    }

    /// Declares a named result with derived provenance.
    pub fn result(self, name: &str, ty: Type) -> Self {
        self.result_with(name, ty, Provenance::derived())
    }

    /// Declares a named result with explicit provenance.
    pub fn result_with(mut self, name: &str, ty: Type, provenance: Provenance) -> Self {
        self.results.push((Some(name.to_string()), ty, provenance));
        self
    }

    /// Sets an attribute.
    pub fn attr(mut self, key: &str, value: impl Into<Attribute>) -> Self {
        self.attributes.insert(key.to_string(), value.into());
        self
    }

    /// Sets the declared effect (§2.2).
    pub fn effect(mut self, effect: Effect) -> Self {
        self.effect = effect;
        self
    }

    /// Inserts the operation and returns its id.
    pub fn build(self) -> OperationId {
        let OpBuilder { builder, name, literal, operands, results, attributes, effect } = self;
        let op = builder
            .module
            .create_op(name, literal, operands, results, attributes, effect);
        let block = builder.insertion;
        builder.module.append_to_block(block, op);
        op
    }

    /// Inserts the operation and returns its single result.
    ///
    /// # Panics
    ///
    /// Panics unless exactly one result was declared.
    pub fn build_one(self) -> ValueId {
        self.build_pair().1
    }

    /// Inserts the operation and returns both its id and its single result.
    ///
    /// # Panics
    ///
    /// Panics unless exactly one result was declared.
    pub fn build_pair(self) -> (OperationId, ValueId) {
        assert_eq!(
            self.results.len(),
            1,
            "build_one/build_pair need exactly one declared result"
        );
        let OpBuilder { builder, name, literal, operands, results, attributes, effect } = self;
        let op = builder
            .module
            .create_op(name, literal, operands, results, attributes, effect);
        let block = builder.insertion;
        builder.module.append_to_block(block, op);
        let result = builder.module.op(op).results[0];
        (op, result)
    }
}
