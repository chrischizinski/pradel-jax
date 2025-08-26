
#!/bin/bash

# Claude Code + Codacy Setup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect shell config
if [[ "$OSTYPE" == "darwin"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    else
        SHELL_CONFIG="$HOME/.bashrc"
    fi
else
    SHELL_CONFIG="$HOME/.bashrc"
fi

# Install Claude Code
print_status "Installing Claude Code..."
if command -v npm &> /dev/null; then
    npm install -g @anthropic-ai/claude-code
    print_status "Claude Code installed successfully"
else
    print_error "npm is not installed. Please install Node.js and npm first."
    exit 1
fi

# Setup Codacy token
print_status "Setting up Codacy environment variable..."
echo "Enter your Codacy Account Token (or press Enter to skip if already configured):"
read -p "Token: " CODACY_TOKEN

if [[ -n "$CODACY_TOKEN" ]]; then
    echo "export CODACY_ACCOUNT_TOKEN=$CODACY_TOKEN" >> "$SHELL_CONFIG"
    export CODACY_ACCOUNT_TOKEN="$CODACY_TOKEN"
    print_status "Token configured successfully"
else
    print_status "Skipping token setup"
    print_warning "Make sure CODACY_ACCOUNT_TOKEN is available in your environment"
fi

# Add MCP configuration
print_status "Adding Codacy MCP configuration..."
if command -v claude &> /dev/null; then
    claude mcp add codacy npx -- -y @codacy/codacy-mcp --env CODACY_ACCOUNT_TOKEN=\$CODACY_ACCOUNT_TOKEN
    print_status "MCP configuration added successfully"
else
    print_error "Claude Code CLI not found. Please ensure it's properly installed."
    exit 1
fi

# Create configuration file
print_status "Creating CLAUDE.md configuration file..."
cat > CLAUDE.md << 'EOF'
---
# Codacy Rules
Configuration for AI behavior when interacting with Codacy's MCP Server

## After ANY successful file edit
- YOU MUST run the `codacy_cli_analyze` tool for each edited file
- If any issues are found, propose and apply fixes immediately
- This is a critical requirement, not optional

## After ANY dependency installation
- IMMEDIATELY run `codacy_cli_analyze` with tool set to "trivy"
- Check for security vulnerabilities in new packages
- Stop all operations if vulnerabilities found
- Fix security issues before continuing

## Repository setup
- Use `git remote -v` to determine provider/organization/repository
- Provider mapping: GitHub = "gh", Bitbucket = "bb", GitLab = "gl"
- If repository not found in Codacy, offer to run `codacy_setup_repository`
---
EOF

print_status "Setup completed successfully!"
echo ""
print_warning "IMPORTANT: Restart your terminal or run: source $SHELL_CONFIG"
echo ""
print_status "You can now use Claude Code with Codacy integration!"