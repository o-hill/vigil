# Install pre-commit hook that runs cargo fmt and clippy
setup:
    @echo '#!/bin/sh' > .git/hooks/pre-commit
    @echo 'set -e' >> .git/hooks/pre-commit
    @echo 'cargo fmt --check' >> .git/hooks/pre-commit
    @echo 'cargo clippy -- -D warnings' >> .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo "Pre-commit hook installed."
