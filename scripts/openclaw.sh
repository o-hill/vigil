#!/usr/bin/env bash
set -euo pipefail

OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"
GATEWAY_PORT="${OPENCLAW_PORT:-18789}"

# ── Helpers ──────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] <command>

Commands:
  install     Install OpenClaw (and run onboarding)
  run         Start the OpenClaw gateway in the foreground
  setup       install + run in one step

Options:
  --wipe      Remove existing OpenClaw installation before installing
  --port N    Gateway port (default: 18789, or \$OPENCLAW_PORT)
  -h, --help  Show this help message
EOF
    exit 0
}

info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
die()   { error "$@"; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || die "'$1' is required but not found in PATH."
}

# ── Actions ──────────────────────────────────────────────────────────────────

wipe() {
    info "Wiping existing OpenClaw installation..."

    # Uninstall the global npm package (ignore errors if not installed).
    if command -v openclaw &>/dev/null; then
        info "Uninstalling openclaw npm package..."
        npm uninstall -g openclaw 2>/dev/null || true
    fi

    # Stop daemon if running.
    if command -v openclaw &>/dev/null; then
        openclaw gateway stop 2>/dev/null || true
    fi

    # Remove data directory.
    if [[ -d "$OPENCLAW_HOME" ]]; then
        info "Removing $OPENCLAW_HOME..."
        rm -rf "$OPENCLAW_HOME"
    fi

    info "Wipe complete."
}

check_node() {
    require_cmd node
    local node_major
    node_major=$(node --version | sed 's/^v//' | cut -d. -f1)
    if (( node_major < 22 )); then
        die "Node.js >= 22 is required (found v$(node --version | sed 's/^v//')). Install a newer version and try again."
    fi
}

install_openclaw() {
    info "Installing OpenClaw..."
    check_node
    require_cmd npm
    npm install -g openclaw@latest

    info "Running OpenClaw onboarding..."
    openclaw onboard --install-daemon
    info "Install complete."
}

run_openclaw() {
    require_cmd openclaw
    info "Starting OpenClaw gateway on port $GATEWAY_PORT (foreground, verbose)..."
    exec openclaw gateway --port "$GATEWAY_PORT" --verbose
}

# ── CLI Parsing ──────────────────────────────────────────────────────────────

DO_WIPE=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wipe)    DO_WIPE=true; shift ;;
        --port)    GATEWAY_PORT="$2"; shift 2 ;;
        -h|--help) usage ;;
        install|run|setup)
            [[ -z "$COMMAND" ]] || die "Multiple commands specified."
            COMMAND="$1"; shift ;;
        *)         die "Unknown argument: $1" ;;
    esac
done

[[ -n "$COMMAND" ]] || { error "No command specified."; usage; }

# ── Execute ──────────────────────────────────────────────────────────────────

case "$COMMAND" in
    install)
        $DO_WIPE && wipe
        install_openclaw
        ;;
    run)
        run_openclaw
        ;;
    setup)
        $DO_WIPE && wipe
        install_openclaw
        run_openclaw
        ;;
esac
