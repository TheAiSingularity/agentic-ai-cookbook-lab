"""engine.mcp — Model Context Protocol server + Claude plugin bundle.

`server.py` exposes the full research pipeline as a single MCP tool called
`research`. Any MCP-speaking client (Claude Desktop, Cursor, Continue,
custom agents) can invoke it over stdio with no additional setup.

The `claude_plugin/` directory is a ready-to-submit bundle for the
Anthropic plugin marketplace — it contains `.claude-plugin/plugin.json`,
four Claude skills, and agent definitions that compose the MCP server
into useful workflows.
"""
