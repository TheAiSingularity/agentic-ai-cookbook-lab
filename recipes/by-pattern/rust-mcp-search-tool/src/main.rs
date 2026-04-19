//! `mcp-search` — a tiny MCP server exposing a single `search` tool backed
//! by a self-hosted SearXNG instance. Companion to the Python pipeline; see
//! the recipe's README for the benchmark numbers justifying its existence.
//!
//! Protocol: stdio transport (the MCP default); tool surface: one tool named
//! `search` taking `{query: string, num_results?: int}` and returning a JSON
//! array of `{url, title, snippet}` objects.
//!
//! Configure the SearXNG endpoint via `SEARXNG_URL` (default `http://localhost:8888`).

use anyhow::{Context, Result};
use reqwest::Client;
use rmcp::{
    handler::server::tool::{Parameters, ToolRouter},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
    ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const DEFAULT_SEARXNG_URL: &str = "http://localhost:8888";
const DEFAULT_NUM_RESULTS: u32 = 5;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
struct SearchArgs {
    /// The search query to dispatch to SearXNG.
    query: String,
    /// How many results to return. Default 5, max 20.
    #[serde(default)]
    num_results: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Hit {
    url: String,
    title: String,
    snippet: String,
}

#[derive(Debug, Deserialize)]
struct SearxngResponse {
    #[serde(default)]
    results: Vec<SearxngHit>,
}

#[derive(Debug, Deserialize)]
struct SearxngHit {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    content: String,
}

#[derive(Clone)]
struct SearchService {
    http: Client,
    base_url: Arc<String>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl SearchService {
    fn new() -> Self {
        let base_url = std::env::var("SEARXNG_URL").unwrap_or_else(|_| DEFAULT_SEARXNG_URL.into());
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .expect("reqwest client"),
            base_url: Arc::new(base_url),
            tool_router: Self::tool_router(),
        }
    }

    /// Search the web via SearXNG and return the top hits.
    ///
    /// Returns a JSON array of `{url, title, snippet}`. Timeouts at 20s.
    #[tool(description = "Search the web via SearXNG. Returns an array of {url, title, snippet}.")]
    async fn search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let n = args.num_results.unwrap_or(DEFAULT_NUM_RESULTS).min(20);
        let url = format!("{}/search", self.base_url);
        let resp = self
            .http
            .get(&url)
            .query(&[("q", args.query.as_str()), ("format", "json")])
            .send()
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(format!("searxng request: {e}"), None))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(rmcp::ErrorData::internal_error(
                format!("searxng returned {status}"),
                None,
            ));
        }

        let body: SearxngResponse = resp
            .json()
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(format!("searxng json: {e}"), None))?;

        let hits: Vec<Hit> = body
            .results
            .into_iter()
            .take(n as usize)
            .map(|h| Hit {
                url: h.url,
                title: h.title,
                snippet: h.content,
            })
            .collect();

        let json = serde_json::to_string(&hits)
            .map_err(|e| rmcp::ErrorData::internal_error(format!("serialize: {e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler]
impl ServerHandler for SearchService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "mcp-search".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                ..Default::default()
            },
            instructions: Some(
                "Search the web via SearXNG. One tool: `search`. Configure SEARXNG_URL to point at your instance.".into(),
            ),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let service = SearchService::new();
    tracing::info!(
        base_url = %service.base_url,
        "mcp-search starting on stdio transport"
    );

    let server = service
        .serve(stdio())
        .await
        .context("failed to start MCP server on stdio")?;

    server.waiting().await?;
    Ok(())
}
