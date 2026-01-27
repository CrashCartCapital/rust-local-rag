#[test]
fn test_prd_t2_1_model_prefix_exists() {
    let tags = serde_json::json!({
        "models": [
            { "name": "nomic-embed-text:latest" },
            { "name": "dengcao/Qwen3-Reranker-4B:Q5_K_M" }
        ]
    });

    assert!(rust_local_rag::doctor::model_prefix_exists(
        &tags,
        "nomic-embed-text"
    ));
    assert!(rust_local_rag::doctor::model_prefix_exists(
        &tags,
        "dengcao/Qwen3-Reranker-4B"
    ));
    assert!(!rust_local_rag::doctor::model_prefix_exists(
        &tags,
        "missing-model"
    ));
}
