//! Tokenizer for the Agent IR textual syntax.

use std::fmt;

/// A position in the source, for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span {
    /// 1-based line.
    pub line: u32,
    /// 1-based column, in characters.
    pub column: u32,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A lexical token.
#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    /// A bare identifier or keyword.
    Ident(String),
    /// `%name`.
    Value(String),
    /// `@name`.
    Symbol(String),
    /// `^name`.
    Label(String),
    /// `#name`.
    Effect(String),
    /// `!dialect.name`.
    TypeName(String),
    /// An integer literal.
    Int(i64),
    /// A floating point literal.
    Float(f64),
    /// A string literal, already unescaped.
    Str(String),
    /// One of `{ } ( ) [ ] < > : , = * .`
    Punct(char),
    /// End of input.
    Eof,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Ident(s) => write!(f, "`{s}`"),
            Tok::Value(s) => write!(f, "`%{s}`"),
            Tok::Symbol(s) => write!(f, "`@{s}`"),
            Tok::Label(s) => write!(f, "`^{s}`"),
            Tok::Effect(s) => write!(f, "`#{s}`"),
            Tok::TypeName(s) => write!(f, "`!{s}`"),
            Tok::Int(v) => write!(f, "`{v}`"),
            Tok::Float(v) => write!(f, "`{v}`"),
            Tok::Str(s) => write!(f, "string `{s}`"),
            Tok::Punct(c) => write!(f, "`{c}`"),
            Tok::Eof => f.write_str("end of input"),
        }
    }
}

/// A token together with where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// What was read.
    pub tok: Tok,
    /// Where it was read.
    pub span: Span,
}

/// A lexing failure.
#[derive(Clone, Debug, PartialEq)]
pub struct LexError {
    /// What went wrong.
    pub message: String,
    /// Where it went wrong.
    pub span: Span,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.span, self.message)
    }
}

impl std::error::Error for LexError {}

/// Turns source text into a token stream, ending with [`Tok::Eof`].
pub fn tokenize(source: &str) -> Result<Vec<Token>, LexError> {
    Lexer::new(source).run()
}

struct Lexer<'a> {
    chars: Vec<char>,
    pos: usize,
    line: u32,
    column: u32,
    source: std::marker::PhantomData<&'a str>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Lexer {
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            source: std::marker::PhantomData,
        }
    }

    fn span(&self) -> Span {
        Span { line: self.line, column: self.column }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn error(&self, message: impl Into<String>, span: Span) -> LexError {
        LexError { message: message.into(), span }
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('/') if self.peek_at(1) == Some('/') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => break,
            }
        }
    }

    fn is_ident_start(c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }

    fn is_ident_continue(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    fn read_ident(&mut self) -> String {
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if Self::is_ident_continue(c) {
                out.push(c);
                self.bump();
            } else {
                break;
            }
        }
        out
    }

    /// Reads a dotted name, as used by type names: `core.tensor`.
    fn read_dotted(&mut self) -> String {
        let mut out = self.read_ident();
        while self.peek() == Some('.')
            && self.peek_at(1).is_some_and(Self::is_ident_start)
        {
            self.bump();
            out.push('.');
            out.push_str(&self.read_ident());
        }
        out
    }

    fn read_string(&mut self, span: Span) -> Result<String, LexError> {
        self.bump(); // opening quote
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(self.error("unterminated string literal", span)),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(other) => {
                        return Err(self.error(format!("unknown escape `\\{other}`"), span))
                    }
                    None => return Err(self.error("unterminated escape", span)),
                },
                Some(c) => out.push(c),
            }
        }
    }

    fn read_number(&mut self, span: Span) -> Result<Tok, LexError> {
        let mut text = String::new();
        if self.peek() == Some('-') {
            text.push('-');
            self.bump();
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                text.push(c);
                self.bump();
            } else {
                break;
            }
        }
        let mut is_float = false;
        // A `.` only continues the number when a digit follows, so `1.foo`
        // cannot swallow the dot of a dotted name.
        if self.peek() == Some('.') && self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            text.push('.');
            self.bump();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    text.push(c);
                    self.bump();
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let save = (self.pos, self.line, self.column);
            let mut exponent = String::new();
            exponent.push(self.bump().unwrap());
            if matches!(self.peek(), Some('+' | '-')) {
                exponent.push(self.bump().unwrap());
            }
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        exponent.push(c);
                        self.bump();
                    } else {
                        break;
                    }
                }
                is_float = true;
                text.push_str(&exponent);
            } else {
                // Not an exponent after all: `1e` is an int followed by a name.
                self.pos = save.0;
                self.line = save.1;
                self.column = save.2;
            }
        }

        if is_float {
            text.parse::<f64>()
                .map(Tok::Float)
                .map_err(|_| self.error(format!("`{text}` is not a valid float"), span))
        } else {
            text.parse::<i64>()
                .map(Tok::Int)
                .map_err(|_| self.error(format!("`{text}` does not fit in an i64"), span))
        }
    }

    fn run(mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_trivia();
            let span = self.span();
            let Some(c) = self.peek() else {
                tokens.push(Token { tok: Tok::Eof, span });
                return Ok(tokens);
            };

            let tok = match c {
                '"' => Tok::Str(self.read_string(span)?),
                '%' => {
                    self.bump();
                    let name = self.read_ident();
                    if name.is_empty() {
                        return Err(self.error("`%` must be followed by a value name", span));
                    }
                    Tok::Value(name)
                }
                '@' => {
                    self.bump();
                    let name = self.read_ident();
                    if name.is_empty() {
                        return Err(self.error("`@` must be followed by a symbol name", span));
                    }
                    Tok::Symbol(name)
                }
                '^' => {
                    self.bump();
                    let name = self.read_ident();
                    if name.is_empty() {
                        return Err(self.error("`^` must be followed by a block label", span));
                    }
                    Tok::Label(name)
                }
                '#' => {
                    self.bump();
                    let name = self.read_ident();
                    if name.is_empty() {
                        return Err(self.error("`#` must be followed by an effect name", span));
                    }
                    Tok::Effect(name)
                }
                '!' => {
                    self.bump();
                    let name = self.read_dotted();
                    if name.is_empty() {
                        return Err(self.error("`!` must be followed by a type name", span));
                    }
                    Tok::TypeName(name)
                }
                c if c.is_ascii_digit() => self.read_number(span)?,
                '-' if self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) => {
                    self.read_number(span)?
                }
                c if Self::is_ident_start(c) => Tok::Ident(self.read_ident()),
                '{' | '}' | '(' | ')' | '[' | ']' | '<' | '>' | ':' | ',' | '=' | '*' | '.' => {
                    self.bump();
                    Tok::Punct(c)
                }
                other => {
                    return Err(self.error(format!("unexpected character `{other}`"), span));
                }
            };
            tokens.push(Token { tok, span });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(source: &str) -> Vec<Tok> {
        tokenize(source)
            .unwrap()
            .into_iter()
            .map(|t| t.tok)
            .filter(|t| *t != Tok::Eof)
            .collect()
    }

    #[test]
    fn lexes_the_sigils() {
        assert_eq!(
            kinds("%v @sym ^bb #pure !core.int"),
            vec![
                Tok::Value("v".into()),
                Tok::Symbol("sym".into()),
                Tok::Label("bb".into()),
                Tok::Effect("pure".into()),
                Tok::TypeName("core.int".into()),
            ]
        );
    }

    #[test]
    fn distinguishes_ints_from_floats() {
        assert_eq!(kinds("1 1.5 -2 -2.5 1e3 1.0e-2"), vec![
            Tok::Int(1),
            Tok::Float(1.5),
            Tok::Int(-2),
            Tok::Float(-2.5),
            Tok::Float(1000.0),
            Tok::Float(0.01),
        ]);
    }

    #[test]
    fn a_trailing_dot_does_not_join_a_number_to_a_name() {
        assert_eq!(
            kinds("1 . agent.func"),
            vec![
                Tok::Int(1),
                Tok::Punct('.'),
                Tok::Ident("agent".into()),
                Tok::Punct('.'),
                Tok::Ident("func".into()),
            ]
        );
    }

    #[test]
    fn unescapes_strings() {
        assert_eq!(kinds(r#""a\nb\"c""#), vec![Tok::Str("a\nb\"c".into())]);
    }

    #[test]
    fn skips_line_comments() {
        assert_eq!(kinds("// gone\n42 // also gone"), vec![Tok::Int(42)]);
    }

    #[test]
    fn reports_where_it_failed() {
        let err = tokenize("ok\n  $").unwrap_err();
        assert_eq!(err.span.line, 2);
        assert_eq!(err.span.column, 3);
    }

    #[test]
    fn an_unterminated_string_is_an_error() {
        assert!(tokenize("\"nope").is_err());
    }
}
