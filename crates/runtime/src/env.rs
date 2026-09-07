//! The boundary between the plan and the world.
//!
//! Everything the runtime cannot do itself goes through [`Environment`]: tool
//! calls, model calls, the memory store. That is the only place an Agent IR
//! program can reach outside, and by the time a call arrives here the verifier
//! has already established that a capability covers it (§7).

use crate::value::Value;
use agent_ir_core::Effect;
use std::collections::BTreeMap;
use std::fmt;

/// One request to the outside world.
#[derive(Clone, Debug, PartialEq)]
pub struct Invocation {
    /// The tool, model or memory key being addressed.
    pub target: String,
    /// The arguments, in operand order.
    pub args: Vec<Value>,
    /// The operation's attributes, for anything the target needs beyond args.
    pub attributes: BTreeMap<String, Value>,
    /// The declared effect. The environment may rely on it: a `ReadExternal`
    /// call is safe to retry, a `WriteExternal` one is not.
    pub effect: Effect,
    /// Present for every non-replayable effect (§8.3).
    pub idempotency_key: Option<String>,
}

impl Invocation {
    /// The first argument, if any.
    pub fn arg(&self, index: usize) -> Option<&Value> {
        self.args.get(index)
    }

    /// One attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&Value> {
        self.attributes.get(name)
    }
}

/// Why a call to the world failed.
#[derive(Clone, Debug, PartialEq)]
pub enum EnvError {
    /// The environment has no such tool or model.
    Unknown(String),
    /// The call was made but failed.
    Failed(String),
    /// The process died. Used by tests to simulate §16's crash, and by a real
    /// environment to report a lost connection.
    Crashed(String),
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvError::Unknown(what) => write!(f, "no such target: {what}"),
            EnvError::Failed(why) => write!(f, "call failed: {why}"),
            EnvError::Crashed(why) => write!(f, "the runtime lost its environment: {why}"),
        }
    }
}

impl std::error::Error for EnvError {}

/// What a plan may ask of the world.
///
/// Every method defaults to [`EnvError::Unknown`], so an environment only
/// implements the capabilities its agent actually holds — a program that tries
/// anything else fails loudly instead of silently doing nothing.
pub trait Environment {
    /// Invokes a tool.
    fn tool_call(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        Err(EnvError::Unknown(format!("tool `{}`", call.target)))
    }

    /// Asks a model.
    fn infer(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        Err(EnvError::Unknown(format!("model `{}`", call.target)))
    }

    /// Reads one memory entry.
    fn memory_read(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        Err(EnvError::Unknown(format!("memory key `{}`", call.target)))
    }

    /// Writes one memory entry.
    fn memory_write(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        Err(EnvError::Unknown(format!("memory key `{}`", call.target)))
    }

    /// Searches memory.
    fn memory_search(&mut self, _call: &Invocation) -> Result<Value, EnvError> {
        Err(EnvError::Unknown("memory search".to_string()))
    }

    /// Whether this environment already recorded a result for an idempotency
    /// key (§8.3).
    ///
    /// The executor keeps its own ledger and consults it first; this hook is
    /// for an environment whose backend tracks keys itself — a payment provider
    /// that deduplicates on a request id, for instance. Returning `None`, the
    /// default, means "I do not track keys, use your own ledger".
    fn lookup_idempotent(&mut self, _key: &str) -> Option<Value> {
        None
    }
}

/// A closure-backed environment that records everything it was asked to do.
///
/// This is what the tests and the `agent-ir run` demo use: register a handler
/// per tool name, then assert on the call log afterwards. The recorded calls
/// are what makes the semantic non-regression tests of §12 phase 5 possible —
/// optimizing a program must not change the sequence of effects it performs.
pub struct RecordingEnvironment {
    handlers: BTreeMap<String, Handler>,
    /// Every call that reached the world, in order.
    pub calls: Vec<Invocation>,
    memory: BTreeMap<String, Value>,
    /// Keys of writes this environment has already applied.
    ///
    /// §8.3 puts this check in the tool runtime rather than only in the
    /// executor's ledger, and the reason shows up in §16: a write whose
    /// acknowledgement is lost to a crash never reaches the executor's ledger,
    /// so only the side that actually performed it can say it happened.
    completed: BTreeMap<String, Value>,
    /// When set, the environment fails with [`EnvError::Crashed`] once this
    /// many *writing* calls have gone through. Used to stage §16's crash.
    pub crash_after_writes: Option<usize>,
    writes: usize,
}

type Handler = Box<dyn FnMut(&Invocation) -> Result<Value, EnvError> + Send>;

impl Default for RecordingEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordingEnvironment {
    /// An environment that knows nothing yet.
    pub fn new() -> Self {
        RecordingEnvironment {
            handlers: BTreeMap::new(),
            calls: Vec::new(),
            memory: BTreeMap::new(),
            completed: BTreeMap::new(),
            crash_after_writes: None,
            writes: 0,
        }
    }

    /// Registers a handler for one tool or model name.
    pub fn on(
        mut self,
        name: &str,
        handler: impl FnMut(&Invocation) -> Result<Value, EnvError> + Send + 'static,
    ) -> Self {
        self.handlers.insert(name.to_string(), Box::new(handler));
        self
    }

    /// Registers a handler that always answers the same thing.
    pub fn returning(self, name: &str, value: Value) -> Self {
        self.on(name, move |_| Ok(value.clone()))
    }

    /// Makes the environment crash after `count` writing calls.
    pub fn crashing_after_writes(mut self, count: usize) -> Self {
        self.crash_after_writes = Some(count);
        self
    }

    /// The names of the targets that were called, in order.
    pub fn call_names(&self) -> Vec<&str> {
        self.calls.iter().map(|c| c.target.as_str()).collect()
    }

    /// Only the calls that changed something outside the program.
    pub fn writing_calls(&self) -> Vec<&Invocation> {
        self.calls.iter().filter(|c| c.effect.is_write()).collect()
    }

    /// The current contents of the memory store.
    pub fn memory(&self) -> &BTreeMap<String, Value> {
        &self.memory
    }

    fn dispatch(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        if call.effect.is_write() {
            self.writes += 1;
            // The write is applied here, so its key is recorded here — before
            // anything can go wrong on the way back.
            if let Some(key) = &call.idempotency_key {
                self.completed.insert(key.clone(), Value::Null);
            }
            if self.crash_after_writes == Some(self.writes) {
                // Exactly like a real write whose acknowledgement is lost: the
                // effect happened, the caller never learned that it did.
                self.calls.push(call.clone());
                return Err(EnvError::Crashed(format!(
                    "simulated crash after {} write(s)",
                    self.writes
                )));
            }
        }
        self.calls.push(call.clone());
        let result = match self.handlers.get_mut(&call.target) {
            Some(handler) => handler(call),
            None => Err(EnvError::Unknown(format!("`{}`", call.target))),
        };
        if let (Ok(value), Some(key)) = (&result, &call.idempotency_key) {
            self.completed.insert(key.clone(), value.clone());
        }
        result
    }
}

impl Environment for RecordingEnvironment {
    fn lookup_idempotent(&mut self, key: &str) -> Option<Value> {
        self.completed.get(key).cloned()
    }

    fn tool_call(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        self.dispatch(call)
    }

    fn infer(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        self.dispatch(call)
    }

    fn memory_read(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        self.calls.push(call.clone());
        Ok(self.memory.get(&call.target).cloned().unwrap_or(Value::Null))
    }

    fn memory_write(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        self.writes += 1;
        if let Some(key) = &call.idempotency_key {
            self.completed.insert(key.clone(), Value::Null);
        }
        if self.crash_after_writes == Some(self.writes) {
            self.calls.push(call.clone());
            self.memory.insert(
                call.target.clone(),
                call.arg(0).cloned().unwrap_or(Value::Null),
            );
            return Err(EnvError::Crashed(format!(
                "simulated crash after {} write(s)",
                self.writes
            )));
        }
        self.calls.push(call.clone());
        self.memory.insert(
            call.target.clone(),
            call.arg(0).cloned().unwrap_or(Value::Null),
        );
        Ok(Value::Null)
    }

    fn memory_search(&mut self, call: &Invocation) -> Result<Value, EnvError> {
        self.calls.push(call.clone());
        Ok(Value::list(self.memory.values().cloned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ir_core::Effect;

    fn invocation(target: &str, effect: Effect) -> Invocation {
        Invocation {
            target: target.to_string(),
            args: vec![Value::Int(1)],
            attributes: BTreeMap::new(),
            effect,
            idempotency_key: None,
        }
    }

    #[test]
    fn an_unregistered_tool_fails_loudly() {
        let mut env = RecordingEnvironment::new();
        let error = env
            .tool_call(&invocation("nope", Effect::read("web")))
            .unwrap_err();
        assert!(matches!(error, EnvError::Unknown(_)));
    }

    #[test]
    fn calls_are_recorded_in_order() {
        let mut env = RecordingEnvironment::new()
            .returning("a", Value::Int(1))
            .returning("b", Value::Int(2));
        env.tool_call(&invocation("a", Effect::read("x"))).unwrap();
        env.tool_call(&invocation("b", Effect::read("x"))).unwrap();
        assert_eq!(env.call_names(), vec!["a", "b"]);
    }

    #[test]
    fn a_crash_still_records_the_write_that_caused_it() {
        // The effect happened; only the acknowledgement was lost. That is
        // precisely the case §8.3 exists for.
        let mut env = RecordingEnvironment::new()
            .returning("pay", Value::Null)
            .crashing_after_writes(1);
        let error = env
            .tool_call(&invocation("pay", Effect::write("ledger")))
            .unwrap_err();
        assert!(matches!(error, EnvError::Crashed(_)));
        assert_eq!(env.writing_calls().len(), 1);
    }

    #[test]
    fn memory_reads_back_what_it_wrote() {
        let mut env = RecordingEnvironment::new();
        let mut write = invocation("k", Effect::write("mem"));
        write.args = vec![Value::Str("v".into())];
        env.memory_write(&write).unwrap();
        let read = env.memory_read(&invocation("k", Effect::read("mem"))).unwrap();
        assert_eq!(read, Value::Str("v".into()));
    }
}
