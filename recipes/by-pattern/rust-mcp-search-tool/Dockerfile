# Multi-stage build — final image is ~8 MB (distroless base + ~5 MB binary).
FROM rust:1.82-slim AS builder
WORKDIR /src
COPY Cargo.toml ./
COPY src ./src
# Pre-create a dummy main so `cargo fetch` caches deps in a separate layer
# (optional optimization; skip if you don't care about image layer cache).
RUN cargo build --release --locked || cargo build --release

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /src/target/release/mcp-search /usr/local/bin/mcp-search
ENV SEARXNG_URL=http://host.docker.internal:8888
ENTRYPOINT ["/usr/local/bin/mcp-search"]
