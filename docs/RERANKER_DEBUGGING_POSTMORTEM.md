# Reranker Debugging Postmortem

## A Forensic Breakdown of Calibrating Phi-4-Mini for RAG Reranking

**Date**: December 7, 2025
**Duration**: ~3 hours of debugging
**Model**: `reranker-phi4mini:latest` (Phi-4-mini, no system prompt)
**Outcome**: Success - reranker now returns meaningful, discriminating scores

---

## 1. The Problem Statement

The rust-local-rag system uses a two-stage retrieval pipeline:
1. **Stage 1**: Embedding similarity search (cosine distance) returns top-k candidates
2. **Stage 2**: LLM-based reranking scores each candidate for relevance to the query

The reranker was consistently logging:
```
No numeric score in reranker response
```

All search results were falling back to embedding similarity scores, meaning Stage 2 was completely non-functional. The reranker was not producing parseable scores.

---

## 2. Initial Hypothesis (Wrong)

**Hypothesis**: The prompt "lacks definitive specific instruction that its response MUST include a numeric score"

**What we tried**: Added explicit instructions demanding a numeric score, added examples, added penalties for non-compliance.

**Result**: Still no parseable scores.

**Why it was wrong**: We were treating this as a prompt engineering problem when it was actually a model format problem. The model wasn't ignoring our instructions - it was never receiving them properly in the first place.

---

## 3. The Debugging Journey (Chronological)

### Phase 1: Prompt Iteration Hell (Wasted Time)

We created multiple prompt versions:
- v1: Basic scoring rubric
- v2: Added penalties for table of contents, prefaces
- v3: Added explicit "YOU MUST INCLUDE A NUMERIC SCORE" instructions

**None of these worked.** We kept getting the same error.

**The stupid mistake**: We kept iterating on the prompt content without ever seeing what the model was actually outputting. We were flying blind.

### Phase 2: System Prompt Conflict Theory

**Hypothesis**: The Ollama model had a baked-in system prompt that was overriding our instructions.

**What we did**: User reinstalled the model WITHOUT a system prompt using a fresh Modelfile.

**Result**: Still broken.

**Lesson**: This was actually the right instinct (system prompts can interfere), but it wasn't the root cause. However, removing the system prompt was necessary - it just wasn't sufficient.

### Phase 3: Adding Debug Logging (First Smart Move)

We added `tracing::debug!` statements to `parse_score()` to see the raw model response.

**Result**: Nothing appeared in logs.

**Why**: The tracing subscriber was filtering out debug-level logs. We were still blind.

### Phase 4: AI Consultation - First Round

We consulted CRASH and Gemini for help. Key insights:

**Codex observation**: "The 'No numeric score' error PROVES parse_score IS being called - the issue is the response content, not whether the function runs."

**Gemini recommendation**: "Use `eprintln!()` to bypass tracing filters entirely."

**This was the first turning point.** Instead of assuming our logging was working, we needed to verify it.

### Phase 5: eprintln Debug Logging (The Breakthrough)

Changed:
```rust
tracing::debug!("Raw response: {}", response);
```
To:
```rust
eprintln!("DEBUG parse_score: Raw response: {}", response);
```

**New problem**: Still no output.

**Why**: We were running the wrong binary.

### Phase 6: The Wrong Binary Problem

**Discovery**: `start-server.sh` runs `rust-local-rag` (installed system binary at `~/.cargo/bin/`), NOT `./target/debug/rust-local-rag` (locally compiled with our debug changes).

We had been:
1. Making code changes
2. Running `cargo build`
3. Running `./start-server.sh`
4. Wondering why nothing changed

**The start-server.sh script doesn't run the debug binary.** This wasted significant time.

**Fix**: Run the debug binary directly:
```bash
source .env && ./target/debug/rust-local-rag
```

### Phase 7: THE RAW OUTPUT REVELATION

Finally seeing the actual model output:

```
DEBUG parse_score: Raw response (first 300 chars): IN ADDITION, YOUR RESPONSE MUST CONTAIN THE FOLLOWING INFORMATION:
- IDENTIFY THE TOPIC OF THE DOCUMENT
- HOW MANY PAGES THE CHUNK COVERS
- THE QUOTE "delta hedging" appears how many times
```

**The model was CONTINUING our prompt text, not answering it.**

Other raw outputs:
```
If the chunk is a preface, foreword, table of contents...
```

The model was in **text completion mode**, not **instruction following mode**.

**This was the core problem all along.** The model wasn't broken, wasn't ignoring instructions - it was literally doing text completion, treating our prompt as incomplete text to continue.

### Phase 8: AI Consultation - Second Round (The Solution Path)

**CRASH analysis**: "phi4-mini without a system prompt acts as a text completion model, not an instruction-following model."

**Gemini's critical insight**:
> "Phi-4-mini requires the Phi chat template with special tokens: `<|user|>...<|end|><|assistant|>`. Without these, it defaults to text completion behavior."

**Gemini's recommended fix**:
1. Use Phi chat template tokens
2. Add one-shot example
3. Pre-fill the response start to force the model into completion mode for the RIGHT thing

### Phase 9: Implementing Phi Template (Partial Success)

Updated prompt to:
```
<|user|>
Score how well this chunk ANSWERS the query (0-100).
[... instructions ...]
Output exactly:
Classification: <DIRECT_ANSWER|PARTIAL_ANSWER|...>
Reasoning: <brief>
Score: <0-100>
<|end|>
<|assistant|>
Classification:
```

**Result**: Model now outputting classifications!
```
DEBUG parse_score: Raw response: Classification: DIRECT_ANSWER
DEBUG parse_score: Raw response: Classification: PARTIAL_ANSWER
```

**New problem**: Model stopped after the Classification line. No Reasoning, no Score.

### Phase 10: The Pre-fill Problem

**Why it happened**: We pre-filled with `Classification:` to force the model to start there. But the model interpreted the Classification line as a complete response and predicted an End-of-Sequence token.

**Gemini's diagnosis**:
> "The pre-fill triggers 'short answer' completion mode. Small models often predict EOS after a single line because they interpret it as complete."

**Gemini's solution**:
> "Switch to JSON format. JSON syntax (opening/closing braces) forces the model to complete all fields before predicting EOS. Pre-fill with `{\"classification\": \"` to enter the JSON structure."

### Phase 11: JSON Format with Pre-fill (Final Solution)

Updated prompt:
```
<|user|>
Score how well this chunk ANSWERS the query (0-100).

[... scoring rubric ...]

== EXAMPLE ==
Query: What is the capital of France?
Document: Geography Textbook
Page: 42
Chunk: Paris is the capital and most populous city of France...
Output: {"classification": "DIRECT_ANSWER", "reasoning": "Explicitly states Paris is the capital", "score": 95}

== YOUR TASK ==
Query: {query}
Document: {document}
Page: {page}
Chunk: {text}

Output:
<|end|>
<|assistant|>
{"classification": "
```

Updated `parse_score()` to reconstruct JSON:
```rust
// Model returns: DIRECT_ANSWER", "reasoning": "...", "score": 95}
// We prepend: {"classification": "
let full_json = format!(r#"{{"classification": "{}"#, response);
// Parse JSON, extract score field
```

Added stop sequences to prevent run-on:
```rust
options: Some(OllamaOptions {
    stop: Some(vec!["<|end|>".to_string(), "<|user|>".to_string()]),
    temperature: Some(0.1),
}),
```

**Result**: SUCCESS!
```
DEBUG parse_score: Raw response: DIRECT_ANSWER", "reasoning": "Directly answers by providing a concise definition", "score": 90}
DEBUG parse_score: Reconstructed JSON: {"classification": "DIRECT_ANSWER", "reasoning": "Directly answers by providing a concise definition", "score": 90}
DEBUG parse_score: Extracted JSON score: 90
```

Search results now show score discrimination:
- Result 1: **0.980** (directly answers)
- Result 2: **0.900** (related)
- Result 3: **0.850** (tangential)

---

## 4. What Was Stupid

1. **Iterating on prompt content without seeing model output**: We made 3+ prompt versions before ever seeing what the model actually returned. Flying blind.

2. **Assuming tracing::debug would work**: Never verified the logging was actually appearing. Should have tested with a simple print statement first.

3. **Running the wrong binary**: We kept rebuilding but running the installed system binary instead of the debug binary. ~30 minutes wasted.

4. **Not questioning the model's mode**: We assumed the model was in instruction-following mode. The fundamental assumption was wrong.

5. **Adding "YOU MUST" instructions**: Adding more emphatic instructions doesn't help when the model isn't even in instruction-following mode. It's like shouting at someone who doesn't speak your language - volume doesn't help.

---

## 5. What Was Smart

1. **AI consultation with specific context**: When we finally consulted Gemini with the actual raw output, it immediately identified the text completion behavior.

2. **Using eprintln() to bypass tracing**: Once suggested, this immediately gave us visibility.

3. **Iterative diagnosis**: CRASH + Gemini consultation after each failure led to progressive understanding.

4. **Removing the system prompt first**: Even though it wasn't the full solution, it was a necessary precondition.

5. **JSON format for structural forcing**: Gemini's insight that JSON braces force complete output was elegant and effective.

---

## 6. The Turning Points

### Turning Point 1: Raw Output Visibility
When we finally saw `"IN ADDITION, YOUR RESPONSE MUST CONTAIN..."`, we immediately knew the model was doing text completion. All previous debugging was in the dark.

### Turning Point 2: Phi Template Recognition
Gemini identifying that Phi-4-mini requires `<|user|>...<|end|><|assistant|>` tokens transformed our understanding. It wasn't a prompt problem - it was a format problem.

### Turning Point 3: JSON Pre-fill Strategy
When the model stopped after Classification, Gemini's suggestion to use JSON format with opening brace pre-fill was the final piece. JSON syntax forces complete output.

---

## 7. The Final Solution (Summary)

### Prompt (`prompts/reranker.txt`):
```
<|user|>
Score how well this chunk ANSWERS the query (0-100).

SCORING:
- 90-100: Directly answers with specific info
- 70-89: Partially answers
- 50-69: Related but doesn't answer
- 25-49: Same topic, not useful
- 0-24: Unrelated

PENALTIES:
- Prefaces/intros describing "what we will cover": MAX 40
- Table of contents, outlines: MAX 30
- Keywords but no answer: MAX 65

== EXAMPLE ==
Query: What is the capital of France?
Document: Geography Textbook
Page: 42
Chunk: Paris is the capital and most populous city of France, with an estimated population of 2.1 million.
Output: {"classification": "DIRECT_ANSWER", "reasoning": "Explicitly states Paris is the capital", "score": 95}

== YOUR TASK ==
Query: {query}
Document: {document}
Page: {page}

Chunk:
{text}

Output:
<|end|>
<|assistant|>
{"classification": "
```

### Code Changes (`src/reranker.rs`):

1. **OllamaOptions struct** with stop sequences:
```rust
#[derive(Serialize)]
struct OllamaOptions {
    stop: Option<Vec<String>>,
    temperature: Option<f32>,
}
```

2. **Request with stop sequences**:
```rust
options: Some(OllamaOptions {
    stop: Some(vec!["<|end|>".to_string(), "<|user|>".to_string()]),
    temperature: Some(0.1),
}),
```

3. **JSON parsing with pre-fill reconstruction**:
```rust
fn parse_score(&self, response: &str) -> Option<f32> {
    // Reconstruct JSON from pre-fill
    let full_json = format!(r#"{{"classification": "{}"#, response);

    // Parse JSON and extract score
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&full_json) {
        if let Some(score) = json.get("score").and_then(|v| v.as_f64()) {
            // Normalize 0-100 to 0.0-1.0
            return Some(if score > 1.0 {
                (score / 100.0).clamp(0.0, 1.0)
            } else {
                score.clamp(0.0, 1.0)
            });
        }
    }

    // Fallback: find "score" followed by number
    // ... text extraction fallback ...
}
```

---

## 8. Lessons for Future LLM Integration

1. **Always verify model output first**: Before any prompt iteration, add temporary logging to see what the model actually returns. `eprintln!` bypasses most logging infrastructure.

2. **Know your model's expected format**: Different models (Phi, Llama, Mistral, etc.) have different chat templates. Phi requires `<|user|>...<|end|><|assistant|>`. Llama uses `[INST]...[/INST]`. Using the wrong format can cause text completion behavior.

3. **Pre-fill strategically**: Ending a prompt with the start of the expected response forces the model into completion mode for that structure. JSON pre-fill (`{"field": "`) is particularly effective because braces require matching.

4. **Stop sequences prevent garbage**: Without stop sequences, models may generate additional turns or continue indefinitely. Always set appropriate stop tokens.

5. **Small models have quirks**: Phi-4-mini often predicts EOS after short responses. JSON format or other structural constraints can force complete output.

6. **System prompts can interfere**: If a model has a baked-in system prompt, it may conflict with your instructions. Test with and without.

7. **Use AI consultants for debugging**: When stuck, describing the exact raw output to another AI (Gemini, Codex) often yields insights you'd miss. They've seen these patterns before.

8. **Verify your test infrastructure**: Make sure you're running the binary you think you're running. Check paths, check which binary `start-server.sh` invokes.

---

## 9. Time Breakdown

| Phase | Time | Value |
|-------|------|-------|
| Prompt iteration without visibility | ~45 min | Wasted |
| System prompt removal | ~10 min | Necessary but not sufficient |
| Debug logging (tracing) | ~15 min | Wasted (wrong approach) |
| Wrong binary debugging | ~30 min | Wasted |
| eprintln debugging | ~5 min | **Critical breakthrough** |
| AI consultation + Phi template | ~20 min | **Solution identified** |
| JSON format implementation | ~15 min | **Solution implemented** |
| Testing and cleanup | ~20 min | Verification |

**Total useful work**: ~60 minutes
**Total wasted work**: ~90 minutes
**Efficiency**: 40%

Most time was wasted by not having visibility into model output. The actual fix, once we understood the problem, took ~35 minutes.

---

## 10. The Meta-Lesson

The debugging process itself was a good example of the value of the AI ensemble approach:

- **Claude (orchestrator)**: Made code changes, ran tests, coordinated
- **CRASH**: Structured the reasoning, tracked hypotheses
- **Gemini**: Provided domain expertise on Phi model behavior, suggested JSON pre-fill
- **Codex**: Confirmed code paths were being executed, validated approach

No single model had the complete picture. The ensemble approach - using multiple AI perspectives on the same problem - accelerated diagnosis significantly once we started using it consistently.

The user's instruction to "continue using CRASH and your consultants regularly for assistance debugging" was the right call. The breakthrough came from consultation, not from isolated iteration.

---

*Document generated from debugging session on December 7, 2025*
