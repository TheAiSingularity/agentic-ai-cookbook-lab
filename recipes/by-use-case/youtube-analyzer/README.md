# youtube-analyzer

**Levels:** beginner ⬛ · production ⬛ · rust ⬛ · **all pending — ships in Wave 1**

## What it does
Given a YouTube URL, produces:
- Transcript (via `yt-dlp` — no YouTube API key required)
- Chapter breakdown with timestamps
- Hierarchical summary (chapter-level → overall)
- Suggested alternative titles (scored)
- Key-quote extraction

All outputs are typed — you get back a Pydantic model you can programmatically consume.

## Who it's for
- Content creators researching topics and auditing their own videos
- Teams summarizing long-form talks, podcasts, or lectures
- Engineers who want a canonical example of typed-output agents

## Why you'd use it
- **Cheapest quality transcript pipeline in 2026:** yt-dlp when subtitles exist (free), Groq Whisper Large v3 Turbo fallback ($0.04/hour, 164–299× real-time)
- **No chunking headaches:** Gemini 2.5 Flash's 1M context handles full transcripts in one call
- **Structured outputs are the whole point** — Pydantic AI's sweet spot
- **Per-video cost: $0.001–$0.02**

## SOTA stack (April 2026)

| Component | Choice | Rationale |
|---|---|---|
| **Orchestration** | Pydantic AI | ~160 LoC for reference chat app (lowest code overhead); typed structured outputs are the differentiator for chapter/title/summary schemas |
| **LLM** | Gemini 2.5 Flash alone | 1M context = entire transcript in one call, zero chunking. $0.25/$1.50 per M tokens. No escalation needed for this task. |
| **Transcript (primary)** | `yt-dlp` | Free; covers ~80% of popular videos via existing subtitles |
| **Transcript (fallback)** | Groq Whisper Large v3 Turbo | Cheapest + fastest ASR April 2026: $0.04/hour, 164–299× real-time |

Pattern: fetch transcript → hierarchical summary (chapter-level schemas → overall) → scored title/description suggestions.

See [`beginner/techniques.md`](beginner/techniques.md) for primary-source citations. *(Lands Wave 1.)*

## Eval

5 test videos × reference outputs. Scorer measures:
- **Chapter-boundary accuracy** — IoU between predicted chapters and reference chapters
- **Summary coverage** — LLM-as-judge: did the summary capture the reference key points?

`make eval` reproduces the score.

## Expected cost per video
$0.001–$0.02 (transcript — often free — plus one Gemini Flash-Lite call).

## See also
- [`../../../foundations/what-is-hermes-agent.md`](../../../foundations/what-is-hermes-agent.md)
