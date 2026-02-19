# ADR 022: Generative Tool Safety Protocols (The Anti-Truncation Law)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Safe File Operations for AI Agents |
| ADR Number          | 022               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## Context
Generative AI tools (like Gemini CLI) interact with the codebase through specific tool calls. A critical failure mode has been identified where the `write_file` tool, when used to "edit" a file by overwriting it, creates a race condition with the tool's output buffer. 

Large files (>100 lines) are frequently silently truncated because the AI model's output token limit or the CLI's standard output buffer cuts off the end of the file content. This results in "Silent Lobotomy"—the file remains valid syntax at the top, but loses critical logic or exports at the bottom, causing cascading system failures that are hard to debug.

## Decision
We strictly enforce a **"Create-Only / Edit-In-Place"** separation of concerns for all Generative AI file operations.

### Overview
1.  **Creation (`write_file`):** May ONLY be used to create **NEW** files. It is strictly prohibited from targeting an existing file path unless the intent is to completely reset it (which must be explicitly confirmed).
2.  **Editing (`replace`):** Must ALWAYS be used for modifying existing files. The AI must locate the specific block to change and apply a targeted replacement.

## Consequences

**Positive Effects:**
- **Zero Truncation Risk:** Targeted replacements do not require re-generating the entire file, thus avoiding token limits and buffer overflows.
- **Context Preservation:** Comments, formatting, and surrounding code in untouched sections remain bit-perfect.
- **Diff Clarity:** Git diffs become smaller and more readable, showing only what changed rather than a full file rewrite.

**Negative Effects:**
- **Complexity:** The AI must accurately identify unique context strings for the `replace` tool, which requires reading the file first. This adds a step to the workflow (Read -> Replace vs. Just Write).

## Rationale
The cost of a single silent truncation event (hours of debugging, potential data loss) far outweighs the cost of the extra "Read" step required for safe editing. This aligns with the project's "Boring Architecture" philosophy (ADR 003): reliability over speed.

### Considerations
*   **ADRs:** This applies specifically to ADRs as well. While ADRs are immutable in spirit, they are occasionally amended (e.g., adding a status update). These amendments must be done via `replace`.

## Additional Notes
This rule is hard-coded into the operational memory of the primary agent.
