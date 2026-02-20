# Agent Development Guidelines

Guidelines for AI agents contributing to this repository.

## Commits

- Small, focused commits. One logical change per commit.
- Short but useful commit messages. Describe *why*, not just *what*.
- Don't bundle unrelated changes together.

## Dependencies

- **Always check that dependencies are the latest stable version** before adding them. Don't copy stale version numbers from training data or prior context — verify against crates.io.
- Minimize dependencies. Prefer the standard library or manual implementations when the added complexity is small. Every dependency is a maintenance and supply-chain burden.
- Justify new dependencies. If a crate saves significant complexity, use it — but don't add one to avoid writing ten lines of code.

## Rust Conventions

- Run `cargo fmt` and `cargo clippy` before committing. Fix all warnings.
- No `unwrap()` or `expect()` in library code. Use proper error handling with `thiserror` and `Result`.
- `unwrap()` is acceptable in tests.
- Use `tracing` for logging, not `println!` or `eprintln!`.

## Testing

- New functionality should include tests.
- Run `cargo test` before committing. All tests must pass.

## Privacy Principle

Vigil never stores parameter values, message content, or tool call results. Only structural metadata. This is a core design invariant — do not introduce code that violates it.
