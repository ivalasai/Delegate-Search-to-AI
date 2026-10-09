# Extraction prompt v1

**Purpose**: Data preparation only. Split each SBIR award abstract into a short
problem statement and a short solution statement. This step is **not** part of
the experimental conditions under test.

**Model**: `phi3:mini` via Ollama (`http://localhost:11434`)

**Temperature**: `0.1` (low; one call per abstract, not repeated)

**Version**: `extraction_prompt_v1`

---

## System prompt

```
You split SBIR award abstracts into two short research statements for a
semantic-matching study. Be faithful to the source text; do not invent facts.
Keep each statement concise (1–3 sentences). Return only valid JSON.
```

## User prompt template

Placeholders: `{award_id}`, `{company}`, `{abstract}`

```
Split this SBIR award abstract into two statements.

Award ID: {award_id}
Company: {company}

Abstract:
{abstract}

Return ONLY a JSON object with exactly these two string fields:
{"problem": "<the need, gap, or challenge being addressed>",
 "solution": "<the proposed technical approach or innovation>"}

Do not include markdown, commentary, or extra keys.
```

## Expected response shape

```json
{
  "problem": "...",
  "solution": "..."
}
```
