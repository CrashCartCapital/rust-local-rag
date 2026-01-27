pub fn model_prefix_exists(tags: &serde_json::Value, prefix: &str) -> bool {
    tags.get("models")
        .and_then(|v| v.as_array())
        .is_some_and(|models| {
            models.iter().any(|m| {
                m.get("name")
                    .and_then(|n| n.as_str())
                    .is_some_and(|name| name.starts_with(prefix))
            })
        })
}

