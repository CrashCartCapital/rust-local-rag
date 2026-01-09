use rmcp::handler::server::common::schema_for_type;

use rust_local_rag::mcp_server::SearchRequest;

fn collect_refs(value: &serde_json::Value, refs: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(r)) = map.get("$ref") {
                refs.push(r.clone());
            }
            for v in map.values() {
                collect_refs(v, refs);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                collect_refs(v, refs);
            }
        }
        _ => {}
    }
}

#[test]
fn search_request_schema_has_no_dangling_refs() {
    let schema = schema_for_type::<SearchRequest>();
    let schema_value = serde_json::Value::Object(schema.clone());

    let definitions = schema.get("definitions").and_then(|v| v.as_object());
    let defs = schema.get("$defs").and_then(|v| v.as_object());

    let mut refs = Vec::new();
    collect_refs(&schema_value, &mut refs);

    for r in refs {
        if let Some(name) = r.strip_prefix("#/definitions/") {
            assert!(
                definitions.and_then(|d| d.get(name)).is_some(),
                "Schema has dangling $ref {r}; missing definitions.{name}"
            );
        } else if let Some(name) = r.strip_prefix("#/$defs/") {
            assert!(
                defs.and_then(|d| d.get(name)).is_some(),
                "Schema has dangling $ref {r}; missing $defs.{name}"
            );
        }
    }
}
