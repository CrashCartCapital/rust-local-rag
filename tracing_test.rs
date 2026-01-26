use tracing::{info, span, Level};
fn main() {
    let span = span!(Level::INFO, "my_span", val = tracing::field::Empty);
    let _enter = span.enter();
    span.record("val", 123);
}
