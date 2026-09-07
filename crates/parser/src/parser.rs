//! Recursive-descent parser for the Agent IR textual syntax.
//!
//! The grammar is exactly what [`agent_ir_core::print_module`] emits:
//!
//! ```text
//! module      := 'module' '@' ident 'version' '(' int ')' '{' capability* operation* '}'
//! capability  := 'capability' '@' ident 'scope' '(' (string | '*') ')'
//!                'grants' '(' ident (',' ident)* ')' 'requires_approval'?
//! operation   := results? op-name string? operands? region* attr-dict? types? provenance?
//! results     := '%' ident (',' '%' ident)* '='
//! operands    := '(' ('%' ident (',' '%' ident)*)? ')'
//! region      := '{' block+ '}'
//! block       := ('^' ident block-args? ':')? operation*
//! attr-dict   := '{' (ident '=' attr (',' ident '=' attr)*)? '}'
//! types       := ':' (type | '(' type (',' type)* ')')
//! provenance  := 'provenance' '(' entry (',' entry)* ')'
//! ```
//!
//! A `{` after the operands is a region unless it looks like an attribute
//! dictionary — that is, unless it is empty or starts with `ident =`. Two
//! tokens of lookahead settle it, and no other production in the grammar needs
//! more.

use crate::lexer::{tokenize, Span, Tok, Token};
use agent_ir_core::{
    Attribute, Attributes, Capability, DType, Effect, EffectClass, Module, OpName, OperationId,
    Provenance, Scope, Source, Type, Validity, ValueId,
};
use std::collections::HashMap;
use std::fmt;

/// A parse failure, with the position that caused it.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    /// What was expected, and what was found.
    pub message: String,
    /// Where the parser gave up.
    pub span: Span,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at {}: {}", self.span, self.message)
    }
}

impl std::error::Error for ParseError {}

impl From<crate::lexer::LexError> for ParseError {
    fn from(err: crate::lexer::LexError) -> Self {
        ParseError {
            message: err.message,
            span: err.span,
        }
    }
}

type Result<T> = std::result::Result<T, ParseError>;

/// Parses a module from its textual form.
pub fn parse_module(source: &str) -> Result<Module> {
    let tokens = tokenize(source)?;
    Parser {
        tokens,
        pos: 0,
        module: Module::new("unnamed"),
        scopes: vec![HashMap::new()],
    }
    .run()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    module: Module,
    scopes: Vec<HashMap<String, ValueId>>,
}

impl Parser {
    // ------------------------------------------------------------ token help

    fn peek(&self) -> &Tok {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].tok
    }

    fn peek_at(&self, offset: usize) -> &Tok {
        &self.tokens[(self.pos + offset).min(self.tokens.len() - 1)].tok
    }

    fn span(&self) -> Span {
        self.tokens[self.pos.min(self.tokens.len() - 1)].span
    }

    fn bump(&mut self) -> Tok {
        let tok = self.tokens[self.pos.min(self.tokens.len() - 1)].tok.clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(ParseError {
            message: message.into(),
            span: self.span(),
        })
    }

    fn at_punct(&self, c: char) -> bool {
        matches!(self.peek(), Tok::Punct(p) if *p == c)
    }

    fn eat_punct(&mut self, c: char) -> bool {
        if self.at_punct(c) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, c: char) -> Result<()> {
        if self.eat_punct(c) {
            Ok(())
        } else {
            self.error(format!("expected `{c}`, found {}", self.peek()))
        }
    }

    fn at_keyword(&self, keyword: &str) -> bool {
        matches!(self.peek(), Tok::Ident(name) if name == keyword)
    }

    fn eat_keyword(&mut self, keyword: &str) -> bool {
        if self.at_keyword(keyword) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<()> {
        if self.eat_keyword(keyword) {
            Ok(())
        } else {
            self.error(format!("expected `{keyword}`, found {}", self.peek()))
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.bump() {
            Tok::Ident(name) => Ok(name),
            other => {
                self.pos -= 1;
                self.error(format!("expected an identifier, found {other}"))
            }
        }
    }

    fn expect_symbol(&mut self) -> Result<String> {
        match self.bump() {
            Tok::Symbol(name) => Ok(name),
            other => {
                self.pos -= 1;
                self.error(format!("expected a `@symbol`, found {other}"))
            }
        }
    }

    fn expect_int(&mut self) -> Result<i64> {
        match self.bump() {
            Tok::Int(v) => Ok(v),
            other => {
                self.pos -= 1;
                self.error(format!("expected an integer, found {other}"))
            }
        }
    }

    fn expect_string(&mut self) -> Result<String> {
        match self.bump() {
            Tok::Str(v) => Ok(v),
            other => {
                self.pos -= 1;
                self.error(format!("expected a string, found {other}"))
            }
        }
    }

    fn expect_float(&mut self) -> Result<f64> {
        match self.bump() {
            Tok::Float(v) => Ok(v),
            Tok::Int(v) => Ok(v as f64),
            other => {
                self.pos -= 1;
                self.error(format!("expected a number, found {other}"))
            }
        }
    }

    // ----------------------------------------------------------- value scope

    fn define(&mut self, name: String, value: ValueId) {
        self.scopes
            .last_mut()
            .expect("the parser always has a scope")
            .insert(name, value);
    }

    fn lookup(&self, name: &str) -> Option<ValueId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn use_value(&mut self, name: &str) -> Result<ValueId> {
        match self.lookup(name) {
            Some(id) => Ok(id),
            None => self.error(format!(
                "`%{name}` is used before it is defined; every value must have a \
                 dominating producer (invariant I1)"
            )),
        }
    }

    // --------------------------------------------------------------- grammar

    fn run(mut self) -> Result<Module> {
        self.expect_keyword("module")?;
        let name = self.expect_symbol()?;
        self.module.name = name;

        if self.eat_keyword("version") {
            self.expect_punct('(')?;
            let version = self.expect_int()?;
            self.expect_punct(')')?;
            self.module.version = u64::try_from(version).unwrap_or(0);
        }

        self.expect_punct('{')?;

        while self.at_keyword("capability") {
            let capability = self.parse_capability()?;
            self.module.capabilities.grant(capability);
        }

        let body = self.module.body_block();
        while !self.at_punct('}') {
            if matches!(self.peek(), Tok::Eof) {
                return self.error("unexpected end of input: the module is not closed");
            }
            self.parse_operation(body)?;
        }
        self.expect_punct('}')?;

        if !matches!(self.peek(), Tok::Eof) {
            return self.error(format!("unexpected {} after the module", self.peek()));
        }
        Ok(self.module)
    }

    fn parse_capability(&mut self) -> Result<Capability> {
        self.expect_keyword("capability")?;
        let name = self.expect_symbol()?;

        self.expect_keyword("scope")?;
        self.expect_punct('(')?;
        let scope = if self.eat_punct('*') {
            Scope::any()
        } else {
            Scope::named(self.expect_string()?)
        };
        self.expect_punct(')')?;

        self.expect_keyword("grants")?;
        self.expect_punct('(')?;
        let mut grants = Vec::new();
        loop {
            let word = self.expect_ident()?;
            match word.parse::<EffectClass>() {
                Ok(class) => grants.push(class),
                Err(()) => return self.error(format!("`{word}` is not an effect class")),
            }
            if !self.eat_punct(',') {
                break;
            }
        }
        self.expect_punct(')')?;

        let mut capability = Capability::new(name, scope, grants);
        if self.eat_keyword("requires_approval") {
            capability = capability.requiring_approval();
        }
        Ok(capability)
    }

    fn parse_operation(&mut self, block: agent_ir_core::BlockId) -> Result<OperationId> {
        // results
        let mut result_names = Vec::new();
        if let Tok::Value(_) = self.peek() {
            // `%a, %b =` is a result list; `%a` on its own starts nothing, so
            // scan past the commas and look for the `=` before committing.
            let mut lookahead = 0;
            while matches!(self.peek_at(lookahead), Tok::Value(_)) {
                lookahead += 1;
                if !matches!(self.peek_at(lookahead), Tok::Punct(',')) {
                    break;
                }
                lookahead += 1;
            }
            if matches!(self.peek_at(lookahead), Tok::Punct('=')) {
                loop {
                    match self.bump() {
                        Tok::Value(name) => result_names.push(name),
                        other => {
                            self.pos -= 1;
                            return self.error(format!("expected a result name, found {other}"));
                        }
                    }
                    if !self.eat_punct(',') {
                        break;
                    }
                }
                self.expect_punct('=')?;
            }
        }

        // dialect.name
        let dialect = self.expect_ident()?;
        self.expect_punct('.')?;
        let op_name = self.expect_ident()?;
        let name = OpName::new(dialect, op_name);

        // literal
        let literal = match self.peek() {
            Tok::Str(_) => Some(self.expect_string()?),
            _ => None,
        };

        // operands
        let mut operands = Vec::new();
        if self.at_punct('(') {
            self.bump();
            if !self.at_punct(')') {
                loop {
                    match self.bump() {
                        Tok::Value(name) => operands.push(self.use_value(&name)?),
                        other => {
                            self.pos -= 1;
                            return self.error(format!("expected `%value`, found {other}"));
                        }
                    }
                    if !self.eat_punct(',') {
                        break;
                    }
                }
            }
            self.expect_punct(')')?;
        }

        // The operation is created before its regions, since a region records
        // the operation that owns it. Result types and the effect are patched
        // in below, once the trailing clauses have been read.
        let results: Vec<(Option<String>, Type, Provenance)> = result_names
            .iter()
            .map(|name| (Some(name.clone()), Type::Unknown, Provenance::derived()))
            .collect();
        let op = self.module.create_op(
            name,
            literal,
            operands,
            results,
            Attributes::new(),
            Effect::Pure,
        );
        self.module.append_to_block(block, op);

        // regions
        while self.at_punct('{') && !self.looks_like_attr_dict() {
            self.parse_region(op)?;
        }

        // attributes (the effect lives among them)
        if self.at_punct('{') {
            let (attributes, effect) = self.parse_attr_dict()?;
            let operation = self.module.op_mut(op);
            operation.attributes = attributes;
            if let Some(effect) = effect {
                operation.effect = effect;
            }
        }

        // result types
        if self.eat_punct(':') {
            let types = if self.eat_punct('(') {
                let mut types = Vec::new();
                if !self.at_punct(')') {
                    loop {
                        types.push(self.parse_type()?);
                        if !self.eat_punct(',') {
                            break;
                        }
                    }
                }
                self.expect_punct(')')?;
                types
            } else {
                vec![self.parse_type()?]
            };
            let result_ids = self.module.op(op).results.clone();
            if types.len() != result_ids.len() {
                return self.error(format!(
                    "the operation declares {} result(s) but {} type(s)",
                    result_ids.len(),
                    types.len()
                ));
            }
            for (id, ty) in result_ids.iter().zip(types) {
                self.module.value_mut(*id).ty = ty;
            }
        } else if !result_names.is_empty() {
            return self.error(format!(
                "`{}` produces {} result(s) and must state their types after `:`",
                self.module.op(op).name,
                result_names.len()
            ));
        }

        // provenance
        if self.at_keyword("provenance") {
            self.parse_provenance(op)?;
        }

        // Results become visible only after the whole operation is read, which
        // is what makes `%x = op(%x)` the error it should be.
        let result_ids = self.module.op(op).results.clone();
        for (name, id) in result_names.into_iter().zip(result_ids) {
            self.define(name, id);
        }

        Ok(op)
    }

    /// Whether the `{` ahead opens an attribute dictionary rather than a region.
    fn looks_like_attr_dict(&self) -> bool {
        matches!(self.peek_at(1), Tok::Punct('}'))
            || (matches!(self.peek_at(1), Tok::Ident(_))
                && matches!(self.peek_at(2), Tok::Punct('=')))
    }

    fn parse_region(&mut self, op: OperationId) -> Result<()> {
        self.expect_punct('{')?;
        let region = self.module.create_op_region(op);
        self.scopes.push(HashMap::new());

        if matches!(self.peek(), Tok::Label(_)) {
            while matches!(self.peek(), Tok::Label(_)) {
                let block = self.module.create_block(region);
                self.bump(); // the label; block names are regenerated on print
                if self.eat_punct('(') {
                    if !self.at_punct(')') {
                        loop {
                            let name = match self.bump() {
                                Tok::Value(name) => name,
                                other => {
                                    self.pos -= 1;
                                    return self.error(format!(
                                        "expected a block argument, found {other}"
                                    ));
                                }
                            };
                            self.expect_punct(':')?;
                            let ty = self.parse_type()?;
                            let value = self.module.add_block_arg(
                                block,
                                Some(name.clone()),
                                ty,
                                Provenance::from_tool(),
                            );
                            self.define(name, value);
                            if !self.eat_punct(',') {
                                break;
                            }
                        }
                    }
                    self.expect_punct(')')?;
                }
                self.expect_punct(':')?;
                while !self.at_punct('}') && !matches!(self.peek(), Tok::Label(_)) {
                    if matches!(self.peek(), Tok::Eof) {
                        return self.error("unexpected end of input inside a region");
                    }
                    self.parse_operation(block)?;
                }
            }
        } else {
            let block = self.module.create_block(region);
            while !self.at_punct('}') {
                if matches!(self.peek(), Tok::Eof) {
                    return self.error("unexpected end of input inside a region");
                }
                self.parse_operation(block)?;
            }
        }

        self.scopes.pop();
        self.expect_punct('}')?;
        Ok(())
    }

    fn parse_attr_dict(&mut self) -> Result<(Attributes, Option<Effect>)> {
        self.expect_punct('{')?;
        let mut attributes = Attributes::new();
        let mut effect = None;
        if !self.at_punct('}') {
            loop {
                let key = self.expect_ident()?;
                self.expect_punct('=')?;
                let value = self.parse_attr_value()?;
                if key == "effect" {
                    match value {
                        Attribute::Effect(e) => effect = Some(e),
                        other => {
                            return self.error(format!(
                                "`effect` must be an effect literal, found {other}"
                            ))
                        }
                    }
                } else {
                    attributes.insert(key, value);
                }
                if !self.eat_punct(',') {
                    break;
                }
            }
        }
        self.expect_punct('}')?;
        Ok((attributes, effect))
    }

    fn parse_attr_value(&mut self) -> Result<Attribute> {
        match self.peek().clone() {
            Tok::Int(v) => {
                self.bump();
                Ok(Attribute::Int(v))
            }
            Tok::Float(v) => {
                self.bump();
                Ok(Attribute::Float(v))
            }
            Tok::Str(v) => {
                self.bump();
                Ok(Attribute::Str(v))
            }
            Tok::Ident(word) if word == "true" || word == "false" => {
                self.bump();
                Ok(Attribute::Bool(word == "true"))
            }
            Tok::Effect(_) => Ok(Attribute::Effect(self.parse_effect()?)),
            Tok::Punct('[') => {
                self.bump();
                let mut items = Vec::new();
                if !self.at_punct(']') {
                    loop {
                        items.push(self.parse_attr_value()?);
                        if !self.eat_punct(',') {
                            break;
                        }
                    }
                }
                self.expect_punct(']')?;
                Ok(Attribute::Array(items))
            }
            Tok::Punct('{') => {
                let (attributes, effect) = self.parse_attr_dict()?;
                let mut entries = attributes;
                if let Some(effect) = effect {
                    entries.insert("effect".to_string(), Attribute::Effect(effect));
                }
                Ok(Attribute::Dict(entries))
            }
            other => self.error(format!("expected an attribute value, found {other}")),
        }
    }

    fn parse_effect(&mut self) -> Result<Effect> {
        let name = match self.bump() {
            Tok::Effect(name) => name,
            other => {
                self.pos -= 1;
                return self.error(format!("expected an effect literal, found {other}"));
            }
        };
        let class: EffectClass = match name.parse() {
            Ok(class) => class,
            Err(()) => return self.error(format!("`#{name}` is not one of the five effects")),
        };
        let scope = if self.eat_punct('<') {
            let scope = Scope::named(self.expect_ident()?);
            self.expect_punct('>')?;
            scope
        } else {
            Scope::any()
        };
        Ok(match class {
            EffectClass::Pure => Effect::Pure,
            EffectClass::Stochastic => Effect::Stochastic,
            EffectClass::ReadExternal => Effect::ReadExternal(scope),
            EffectClass::WriteExternal => Effect::WriteExternal(scope),
            EffectClass::Irreversible => Effect::Irreversible(scope),
        })
    }

    fn parse_type(&mut self) -> Result<Type> {
        let name = match self.bump() {
            Tok::TypeName(name) => name,
            other => {
                self.pos -= 1;
                return self.error(format!("expected a `!type`, found {other}"));
            }
        };
        match name.as_str() {
            "core.int" => Ok(Type::Int),
            "core.float" => Ok(Type::Float),
            "core.bool" => Ok(Type::Bool),
            "core.string" => Ok(Type::Str),
            "core.unknown" => Ok(Type::Unknown),
            "agent.plan" => Ok(Type::Plan),
            "memory.memory" => Ok(Type::Memory),
            "core.tensor" => {
                self.expect_punct('<')?;
                let mut shape = Vec::new();
                loop {
                    match self.peek().clone() {
                        Tok::Int(dim) => {
                            self.bump();
                            shape.push(dim);
                            self.expect_punct(',')?;
                        }
                        Tok::Ident(word) => {
                            self.bump();
                            let dtype = match word.as_str() {
                                "i32" => DType::I32,
                                "i64" => DType::I64,
                                "f32" => DType::F32,
                                "f64" => DType::F64,
                                "i1" => DType::Bool,
                                other => {
                                    return self.error(format!("`{other}` is not an element type"))
                                }
                            };
                            self.expect_punct('>')?;
                            return Ok(Type::Tensor { shape, dtype });
                        }
                        other => {
                            return self.error(format!(
                                "expected a dimension or an element type, found {other}"
                            ))
                        }
                    }
                }
            }
            "tool.ref" => {
                let inner = self.parse_type_argument()?;
                Ok(Type::reference(inner))
            }
            "tool.result" => {
                let inner = self.parse_type_argument()?;
                Ok(Type::tool_result(inner))
            }
            "observation.observation" => {
                let inner = self.parse_type_argument()?;
                Ok(Type::observation(inner))
            }
            other => self.error(format!("`!{other}` is not a type of §2.1")),
        }
    }

    fn parse_type_argument(&mut self) -> Result<String> {
        self.expect_punct('<')?;
        let inner = self.expect_ident()?;
        self.expect_punct('>')?;
        Ok(inner)
    }

    fn parse_provenance(&mut self, op: OperationId) -> Result<()> {
        self.expect_keyword("provenance")?;
        self.expect_punct('(')?;
        loop {
            let name = match self.bump() {
                Tok::Value(name) => name,
                other => {
                    self.pos -= 1;
                    return self.error(format!("expected a `%result`, found {other}"));
                }
            };
            self.expect_punct('=')?;
            let source_word = self.expect_ident()?;
            let source: Source = match source_word.parse() {
                Ok(source) => source,
                Err(()) => {
                    return self.error(format!("`{source_word}` is not a provenance source"))
                }
            };

            let mut confidence = 1.0;
            if self.eat_keyword("confidence") {
                self.expect_punct('(')?;
                confidence = self.expect_float()?;
                self.expect_punct(')')?;
            }

            let mut validity = Validity::Valid;
            if let Tok::Ident(word) = self.peek().clone() {
                if let Ok(parsed) = word.parse::<Validity>() {
                    self.bump();
                    validity = parsed;
                }
            }

            let results = self.module.op(op).results.clone();
            let target = results
                .iter()
                .find(|&&id| self.module.value(id).name.as_deref() == Some(name.as_str()));
            match target {
                Some(&id) => {
                    let provenance = &mut self.module.value_mut(id).provenance;
                    provenance.source = source;
                    provenance.confidence = confidence;
                    provenance.validity = validity;
                }
                None => {
                    return self.error(format!(
                        "`%{name}` is not a result of this operation, so it has no provenance here"
                    ))
                }
            }

            if !self.eat_punct(',') {
                break;
            }
        }
        self.expect_punct(')')?;
        Ok(())
    }
}
