#!/usr/bin/env python3
"""Sync LAMBDA_* environment variables from .env to template.yaml and deploy_stack.sh.

Usage:
    just sync-env
    # or
    uv run python scripts/sync_env.py

This script:
1. Reads .env file and extracts LAMBDA_* prefixed variables
2. Adds CloudFormation Parameters to template.yaml
3. Adds Environment Variables to all Lambda functions in template.yaml
4. Updates deploy_stack.sh to pass the variables as CloudFormation parameters
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse .env file and return LAMBDA_* prefixed variables."""
    if not env_path.exists():
        return {}

    lambda_vars: dict[str, str] = {}
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key.startswith("LAMBDA_"):
                lambda_vars[key] = value

    return lambda_vars


def to_pascal_case(name: str) -> str:
    """Convert LAMBDA_FOO_BAR to FooBar (CloudFormation parameter name)."""
    # Remove LAMBDA_ prefix
    name = name.removeprefix("LAMBDA_")
    # Convert to PascalCase
    return "".join(word.capitalize() for word in name.lower().split("_"))


def to_env_var_name(name: str) -> str:
    """Convert LAMBDA_FOO_BAR to FOO_BAR (Lambda environment variable name)."""
    return name.removeprefix("LAMBDA_")


def update_template_yaml(template_path: Path, lambda_vars: dict[str, str]) -> list[str]:
    """Update template.yaml with new parameters and environment variables.

    Returns list of added variable names.
    """
    if not template_path.exists():
        print(f"Error: {template_path} not found", file=sys.stderr)
        return []

    content = template_path.read_text()
    added: list[str] = []

    for var_name in lambda_vars:
        param_name = to_pascal_case(var_name)
        env_var_name = to_env_var_name(var_name)

        # Check if parameter already exists
        if re.search(rf"^\s*{param_name}:\s*$", content, re.MULTILINE):
            continue

        # Find the Parameters section end (before Resources:)
        # Add new parameter before Resources section
        param_block = f"""  {param_name}:
    Type: String
    NoEcho: true
    Description: "{env_var_name} environment variable (synced from .env)"
"""

        # Insert parameter before Resources:
        if "Resources:" in content:
            content = content.replace(
                "Resources:",
                f"{param_block}\nResources:",
                1,
            )

        # Add environment variable to all Lambda functions
        # Pattern: find Environment:\n        Variables: blocks and add the new var
        def add_env_var(match: re.Match[str]) -> str:
            block = match.group(0)
            indent = match.group(1)
            # Check if this env var already exists in this block
            if f"{env_var_name}:" in block:
                return block
            # Add the new environment variable at the end of Variables block
            # Find the last line of the Variables block
            lines = block.rstrip().split("\n")
            last_line = lines[-1]
            var_indent = " " * (len(last_line) - len(last_line.lstrip()))
            new_var = f"\n{var_indent}{env_var_name}: !Ref {param_name}"
            return block.rstrip() + new_var

        # Match Environment Variables blocks in Lambda functions
        pattern = r"([ ]*)Environment:\s*\n\1  Variables:(?:\n\1    [^\n]+)+"
        content = re.sub(pattern, add_env_var, content)

        added.append(var_name)

    if added:
        template_path.write_text(content)

    return added


def update_deploy_script(script_path: Path, lambda_vars: dict[str, str]) -> list[str]:
    """Update deploy_stack.sh with new parameter overrides.

    Returns list of added variable names.
    """
    if not script_path.exists():
        print(f"Error: {script_path} not found", file=sys.stderr)
        return []

    content = script_path.read_text()
    added: list[str] = []

    # Find the line with PARAM_OVERRIDES assignments
    # We'll add new lines after the last [[ -n "${VAR:-}" ]] && PARAM_OVERRIDES=... line
    lines = content.split("\n")
    new_lines: list[str] = []
    last_param_override_idx = -1

    for i, line in enumerate(lines):
        new_lines.append(line)
        if "PARAM_OVERRIDES=" in line and "&&" in line:
            last_param_override_idx = len(new_lines) - 1

    if last_param_override_idx == -1:
        # Find the initial PARAM_OVERRIDES= line
        for i, line in enumerate(new_lines):
            if line.strip().startswith("PARAM_OVERRIDES=") and "&&" not in line:
                last_param_override_idx = i
                break

    if last_param_override_idx == -1:
        print("Error: Could not find PARAM_OVERRIDES in deploy_stack.sh", file=sys.stderr)
        return []

    # Add new parameter overrides
    insert_lines: list[str] = []
    for var_name in lambda_vars:
        param_name = to_pascal_case(var_name)
        env_var_name = to_env_var_name(var_name)

        # Check if this parameter already exists
        if f"{param_name}=" in content:
            continue

        # Create the new line
        new_line = f'[[ -n "${{{env_var_name}:-}}" ]] && PARAM_OVERRIDES="$PARAM_OVERRIDES {param_name}=${env_var_name}"'
        insert_lines.append(new_line)
        added.append(var_name)

    if insert_lines:
        # Insert after the last PARAM_OVERRIDES line
        for i, line in enumerate(insert_lines):
            new_lines.insert(last_param_override_idx + 1 + i, line)

        script_path.write_text("\n".join(new_lines))

    return added


def main() -> int:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    template_path = project_root / "template.yaml"
    deploy_script_path = project_root / "scripts" / "deploy_stack.sh"

    print("Syncing LAMBDA_* environment variables from .env...")
    print()

    # Parse .env file
    lambda_vars = parse_env_file(env_path)

    if not lambda_vars:
        print("No LAMBDA_* variables found in .env")
        print()
        print("To add a new Lambda environment variable:")
        print("  1. Add LAMBDA_MY_VAR=value to .env")
        print("  2. Run: just sync-env")
        print("  3. Deploy: just deploy")
        return 0

    print(f"Found {len(lambda_vars)} LAMBDA_* variable(s):")
    for var in lambda_vars:
        param = to_pascal_case(var)
        env = to_env_var_name(var)
        print(f"  {var} -> Parameter: {param}, EnvVar: {env}")
    print()

    # Update template.yaml
    template_added = update_template_yaml(template_path, lambda_vars)
    if template_added:
        print(f"Updated template.yaml: added {len(template_added)} parameter(s)")
        for var in template_added:
            print(f"  + {to_pascal_case(var)}")
    else:
        print("template.yaml: no changes needed")

    # Update deploy_stack.sh
    script_added = update_deploy_script(deploy_script_path, lambda_vars)
    if script_added:
        print(f"Updated deploy_stack.sh: added {len(script_added)} parameter override(s)")
        for var in script_added:
            print(f"  + {to_env_var_name(var)}")
    else:
        print("deploy_stack.sh: no changes needed")

    print()
    if template_added or script_added:
        print("Sync complete! Next steps:")
        print("  1. Review changes: git diff")
        print("  2. Set the variable in .env with actual value")
        print("  3. Deploy: just build && just deploy")
    else:
        print("All LAMBDA_* variables are already synced.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
