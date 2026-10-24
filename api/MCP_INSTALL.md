# Full Install Options

This page include more options to install the MCP.  If you use a system not on this list please let us know what you use and how to get it set up if you think others might benefit from that in the future.  

## Testing

Regardless of which method you use, once you have completed the install type into terminal: `I'd like to perforate my model.  Call the Perforated tool to confirm installation and then perform perforation.`.

## Choosing Your Login

To use our MCP server you will need to have a Perforated Token.  For each of the instructions below replace PAI_TOKEN with your actual token and PAI_EMAIL with your email account you used to sign up for that token.

## Cursor

Copy and paste this into your terminal, then restart Cursor

```
python3 -c "
import json, pathlib, os
p = pathlib.Path.home() / '.cursor' / 'mcp.json'
os.makedirs(p.parent, exist_ok=True)
c = json.loads(p.read_text()) if p.exists() and p.stat().st_size > 0 else {}
c.setdefault('mcpServers', {})['perforated'] = {'url': 'https://mcp.perforatedai.app/sse', 'headers': {'Authorization': 'Bearer PAI_EMAIL:PAI_TOKEN'}}
p.write_text(json.dumps(c, indent=2))
"
```

Alternatively open ~/.cursor/mcp.json and add the following content. If the file is not empty, add the perforated section inside the existing "mcpServers" object.  With this method you must also restart Cursor after saving.

```json
{
  "mcpServers": {
    "perforated": {
      "url": "https://mcp.perforatedai.app/sse",
      "headers": {
        "Authorization": "Bearer PAI_EMAIL:PAI_TOKEN"
      }
    }
  }
}
```

## Claude Code

Open a terminal and run this command.

```sh
claude mcp add perforated https://mcp.perforatedai.app/sse \
  --transport http --scope user \
  --header "Authorization: Bearer PAI_EMAIL:PAI_TOKEN"
```

## Claude Desktop

Copy and paste this into your terminal then restart Claude Desktop.

```sh
python3 -c "
import json, pathlib
p = pathlib.Path.home() / 'Library/Application Support/Claude/claude_desktop_config.json'
c = json.loads(p.read_text()) if p.exists() else {}
c.setdefault('mcpServers', {})['perforated'] = {'url': 'https://mcp.perforatedai.app/sse', 'headers': {'Authorization': 'Bearer PAI_EMAIL:PAI_TOKEN'}}
p.write_text(json.dumps(c, indent=2))
print('Done! Restart Claude Desktop to activate.')
"
```

## VS Code

Type Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (Mac), then search for "MCP: Open User Configuration". This opens your global MCP config. Paste this in, save, and restart your coding agent.  If the "servers" field already exists, add perforated rather than making a new one.

```json
{
  "servers": {
    "perforated": {
      "type": "http",
      "url": "https://mcp.perforatedai.app/sse",
      "headers": {
        "Authorization": "Bearer PAI_EMAIL:PAI_TOKEN"
      }
    }
  }
}
```

## Windsurf

Copy and paste this into your terminal then restart Windsurf. To edit later, use Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (Mac) then "Windsurf: Configure MCP Servers".

```sh
python3 -c "
import json, pathlib, os
p = pathlib.Path.home() / '.codeium' / 'windsurf' / 'mcp_config.json'
os.makedirs(p.parent, exist_ok=True)
c = json.loads(p.read_text()) if p.exists() and p.stat().st_size > 0 else {}
c.setdefault('mcpServers', {})['perforated'] = {'serverUrl': 'https://mcp.perforatedai.app/sse', 'headers': {'Authorization': 'Bearer PAI_EMAIL:PAI_TOKEN'}}
p.write_text(json.dumps(c, indent=2))
print('Done! Restart Windsurf to activate.')
"
```

## Codex

Run these commands in your terminal. The first sets your API key as an environment variable, and the second registers the Perforated server with Codex. Add the export line to your shell profile (~/.zshrc or ~/.bashrc) so it persists.

```sh
export PERFORATED_TOKEN="PAI_EMAIL:PAI_TOKEN"

codex mcp add perforated \
  --url https://mcp.perforatedai.app/sse \
  --bearer-token-env-var PERFORATED_TOKEN
```

## Not Listed?

If your agent is not listed then ask it how to add an MCP server and tell it you need the following settings

```text
MCP Server https://mcp.perforatedai.app/sse: https://mcp.perforatedai.app/sse
API Key:        PAI_EMAIL:PAI_TOKEN

Authorization:  Bearer PAI_EMAIL:PAI_TOKEN
```
