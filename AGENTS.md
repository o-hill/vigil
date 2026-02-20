# Agent Development Guidelines

Guidelines for AI agents contributing to this repository.

## Commits

- Small, focused commits. One logical change per commit.
- Short but useful commit messages. Describe *why*, not just *what*.
- Don't bundle unrelated changes together.

## Dependencies

- **Always check that dependencies are the latest stable version** before adding them. Don't copy stale version numbers from training data or prior context — verify against crates.io.
- Justify new dependencies. Prefer the standard library when it's sufficient.

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
