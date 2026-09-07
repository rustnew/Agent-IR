//! The cost model and budgets of §9.1.
//!
//! ```text
//! Cost = α·input_tokens + β·output_tokens + γ·llm_calls + δ·tool_calls + ε·latency
//! ```
//!
//! Every coefficient is a knob rather than a constant, because §0 forbids
//! claiming a number this project has not measured. The defaults below are
//! *placeholders for a benchmark run*, not findings, and the scheduler treats
//! them as such: they order candidate strategies, they never justify a claim.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign};

/// What one execution is expected, or observed, to consume.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Tokens sent to a model.
    pub input_tokens: u64,
    /// Tokens produced by a model.
    pub output_tokens: u64,
    /// How many times a model was asked.
    pub llm_calls: u64,
    /// How many times a tool was invoked.
    pub tool_calls: u64,
    /// Wall-clock milliseconds.
    pub latency_ms: u64,
}

impl Cost {
    /// The zero cost.
    pub const ZERO: Cost = Cost {
        input_tokens: 0,
        output_tokens: 0,
        llm_calls: 0,
        tool_calls: 0,
        latency_ms: 0,
    };

    /// One tool invocation taking `latency_ms`.
    pub fn tool_call(latency_ms: u64) -> Self {
        Cost { tool_calls: 1, latency_ms, ..Cost::ZERO }
    }

    /// One model call.
    pub fn llm_call(input_tokens: u64, output_tokens: u64, latency_ms: u64) -> Self {
        Cost { input_tokens, output_tokens, llm_calls: 1, latency_ms, ..Cost::ZERO }
    }

    /// The total tokens, input and output.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Combines two costs that run one after the other: latency adds up.
    pub fn then(self, next: Cost) -> Cost {
        self + next
    }

    /// Combines two costs that run at the same time: latency is the longer of
    /// the two, everything else still adds up.
    ///
    /// This is the whole benefit *Parallelization* claims, stated precisely
    /// enough to be measured against §9.4 rather than asserted.
    pub fn alongside(self, other: Cost) -> Cost {
        Cost {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            llm_calls: self.llm_calls + other.llm_calls,
            tool_calls: self.tool_calls + other.tool_calls,
            latency_ms: self.latency_ms.max(other.latency_ms),
        }
    }
}

impl Add for Cost {
    type Output = Cost;

    fn add(self, other: Cost) -> Cost {
        Cost {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            llm_calls: self.llm_calls + other.llm_calls,
            tool_calls: self.tool_calls + other.tool_calls,
            latency_ms: self.latency_ms + other.latency_ms,
        }
    }
}

impl AddAssign for Cost {
    fn add_assign(&mut self, other: Cost) {
        *self = *self + other;
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} tokens ({} in / {} out), {} llm call(s), {} tool call(s), {} ms",
            self.total_tokens(),
            self.input_tokens,
            self.output_tokens,
            self.llm_calls,
            self.tool_calls,
            self.latency_ms
        )
    }
}

/// The coefficients of §9.1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostModel {
    /// α — weight on input tokens.
    pub input_token: f64,
    /// β — weight on output tokens.
    pub output_token: f64,
    /// γ — weight on each model call.
    pub llm_call: f64,
    /// δ — weight on each tool call.
    pub tool_call: f64,
    /// ε — weight on each millisecond of latency.
    pub latency_ms: f64,
}

impl CostModel {
    /// Placeholder coefficients, pending the benchmark run of §9.4.
    ///
    /// They are not measurements. They encode only an ordering that is safe to
    /// assume — a model call costs more than a tool call, which costs more than
    /// a token — so the scheduler has something to sort by before any real
    /// numbers exist.
    pub const PLACEHOLDER: CostModel = CostModel {
        input_token: 1.0,
        output_token: 3.0,
        llm_call: 1000.0,
        tool_call: 100.0,
        latency_ms: 0.5,
    };

    /// The scalar the scheduler minimizes.
    pub fn evaluate(&self, cost: &Cost) -> f64 {
        self.input_token * cost.input_tokens as f64
            + self.output_token * cost.output_tokens as f64
            + self.llm_call * cost.llm_calls as f64
            + self.tool_call * cost.tool_calls as f64
            + self.latency_ms * cost.latency_ms as f64
    }
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel::PLACEHOLDER
    }
}

/// The constraints an `agent.budget` operation declares (§9.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Maximum tokens, input plus output.
    pub token_budget: Option<u64>,
    /// Maximum wall-clock milliseconds.
    pub latency_budget_ms: Option<u64>,
    /// Maximum number of model calls.
    pub llm_call_budget: Option<u64>,
    /// Maximum number of tool calls.
    pub tool_call_budget: Option<u64>,
}

/// Why a cost does not fit a budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overrun {
    /// Too many tokens.
    Tokens,
    /// Too slow.
    Latency,
    /// Too many model calls.
    LlmCalls,
    /// Too many tool calls.
    ToolCalls,
}

impl fmt::Display for Overrun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Overrun::Tokens => "token budget",
            Overrun::Latency => "latency budget",
            Overrun::LlmCalls => "model call budget",
            Overrun::ToolCalls => "tool call budget",
        })
    }
}

impl Budget {
    /// No constraints at all.
    pub fn unlimited() -> Self {
        Budget::default()
    }

    /// Whether the budget constrains anything.
    pub fn is_unlimited(&self) -> bool {
        *self == Budget::default()
    }

    /// The first constraint `cost` breaks, if any.
    pub fn overrun(&self, cost: &Cost) -> Option<Overrun> {
        if self.token_budget.is_some_and(|limit| cost.total_tokens() > limit) {
            return Some(Overrun::Tokens);
        }
        if self.latency_budget_ms.is_some_and(|limit| cost.latency_ms > limit) {
            return Some(Overrun::Latency);
        }
        if self.llm_call_budget.is_some_and(|limit| cost.llm_calls > limit) {
            return Some(Overrun::LlmCalls);
        }
        if self.tool_call_budget.is_some_and(|limit| cost.tool_calls > limit) {
            return Some(Overrun::ToolCalls);
        }
        None
    }

    /// Whether the cost fits.
    pub fn admits(&self, cost: &Cost) -> bool {
        self.overrun(cost).is_none()
    }

    /// Reads a budget out of an `agent.budget` operation's attributes.
    pub fn from_attributes(op: &agent_ir_core::Operation) -> Self {
        let read = |key: &str| op.int_attr(key).and_then(|v| u64::try_from(v).ok());
        Budget {
            token_budget: read("token_budget"),
            latency_budget_ms: read("latency_budget_ms"),
            llm_call_budget: read("llm_call_budget"),
            tool_call_budget: read("tool_call_budget"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_latency_adds_and_parallel_latency_maxes() {
        let a = Cost::tool_call(300);
        let b = Cost::tool_call(500);
        assert_eq!(a.then(b).latency_ms, 800);
        assert_eq!(a.alongside(b).latency_ms, 500);
        // The work itself is the same either way.
        assert_eq!(a.then(b).tool_calls, 2);
        assert_eq!(a.alongside(b).tool_calls, 2);
    }

    #[test]
    fn the_model_orders_a_model_call_above_a_tool_call() {
        let model = CostModel::PLACEHOLDER;
        let llm = Cost::llm_call(10, 10, 0);
        let tool = Cost::tool_call(0);
        assert!(model.evaluate(&llm) > model.evaluate(&tool));
    }

    #[test]
    fn an_unlimited_budget_admits_everything() {
        let budget = Budget::unlimited();
        assert!(budget.is_unlimited());
        assert!(budget.admits(&Cost::llm_call(1_000_000, 1_000_000, 999_999)));
    }

    #[test]
    fn each_constraint_reports_itself() {
        let budget = Budget {
            token_budget: Some(100),
            latency_budget_ms: Some(1000),
            llm_call_budget: Some(1),
            tool_call_budget: Some(2),
        };
        assert_eq!(budget.overrun(&Cost::llm_call(200, 0, 0)), Some(Overrun::Tokens));
        assert_eq!(budget.overrun(&Cost::tool_call(2000)), Some(Overrun::Latency));
        assert_eq!(
            budget.overrun(&(Cost::llm_call(1, 1, 1) + Cost::llm_call(1, 1, 1))),
            Some(Overrun::LlmCalls)
        );
        assert_eq!(
            budget.overrun(&(Cost::tool_call(1) + Cost::tool_call(1) + Cost::tool_call(1))),
            Some(Overrun::ToolCalls)
        );
        assert!(budget.admits(&Cost::tool_call(10)));
    }
}
