//! # Agent IR parser
//!
//! Lexer and recursive-descent parser for the textual syntax of §3.3, paired
//! with [`agent_ir_core::print_module`] so that text and IR are two views of
//! the same thing.
//!
//! Round-trip idempotence — `print(parse(print(m))) == print(m)` — is the
//! phase-2 success criterion of §12 and is enforced by this crate's tests
//! against every example in `examples/`.
//!
//! ```
//! let module = agent_ir_parser::parse_module(r#"
//!     module @demo version(0) {
//!       agent.func "main" {
//!         %greeting = core.constant {effect = #pure, value = "hello"} : !core.string
//!         agent.return(%greeting) {effect = #pure}
//!       }
//!     }
//! "#).unwrap();
//!
//! assert_eq!(module.name, "demo");
//! assert!(module.function("main").is_some());
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod lexer;
pub mod parser;

pub use lexer::{tokenize, LexError, Span, Tok, Token};
pub use parser::{parse_module, ParseError};

/// Parses a module, then prints it back: the identity the tests assert on.
pub fn round_trip(source: &str) -> Result<String, ParseError> {
    Ok(agent_ir_core::print_module(&parse_module(source)?))
}
