FROM rust:1.90-bookworm AS build
WORKDIR /source
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 worker
COPY --from=build /source/target/release/attelier-api-miner /usr/local/bin/attelier-api-miner
COPY --chmod=0755 phala-enrollment-bootstrap.sh /usr/local/bin/attelier-api-miner-enroll
USER worker
ENTRYPOINT ["attelier-api-miner"]
