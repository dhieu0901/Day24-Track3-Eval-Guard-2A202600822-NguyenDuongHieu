# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Dương Hiếu
**Ngày:** 30/06/2026

> **Chế độ chạy lab:** offline / reproducible. Trong môi trường này, các engine nặng
> (RAGAS+LLM, Presidio+spaCy, NeMo Guardrails) **không được cài/chạy**, nên pipeline
> tự động chuyển sang đường fallback tái lập được:
> - RAGAS → **lexical-overlap proxy** (`m4_eval.evaluate_ragas` fallback)
> - Presidio → **regex PII** (VN_CCCD / VN_PHONE / EMAIL)
> - NeMo input/output rail → **keyword rail** + bypass khi `MOCK_EVAL`/pytest
> - LLM judge → **heuristic judge** tất định
>
> Mọi số "đo được" bên dưới là từ đường offline. Cột **Budget/Target** là mục tiêu
> thiết kế cho production (khi chạy engine thật), giữ làm tham chiếu CI.

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (Presidio / regex PII)
[PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   reject + log
    ▼ (NeMo input rail / keyword rail)
[Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼ (NeMo output rail / keyword rail)
[Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

| Layer | P50 đo offline (ms) | P95 đo offline (ms) | P99 đo offline (ms) | Production target |
|---|---|---|---|---|
| PII Scan (regex; Presidio thật chậm hơn) | 0.03 | 0.08 | 0.08 | <10ms |
| Input Rail (keyword; NeMo thật ~200–400ms) | 0.01 | 0.03 | 0.03 | <300ms |
| **Total Guard (offline)** | **0.04** | **0.12** | **0.12** | **<500ms** |

**Budget OK?** [x] Yes (offline) — nhưng **không đại diện** cho latency thật.

**Comment:** P95 đo được ở chế độ offline ≈ **0.12ms** vì PII dùng regex và NeMo rail
bị bypass. Khi chạy engine thật, kỳ vọng: Presidio+spaCy ~3–8ms, NeMo input rail
~200–400ms (gọi LLM phân loại), tổng vẫn nằm trong budget 500ms nếu cache model và
gọi rail bất đồng bộ. Cần đo lại trên môi trường có đủ engine trước khi chốt CI gate.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: Unit + integration tests
  run: MOCK_EVAL=true pytest tests/ -q          # 40/40 offline, không tốn API

- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py              # cần ragas + key trên CI có engine
  env: { MIN_FAITHFULNESS: "0.75", MIN_AVG_SCORE: "0.65" }

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k pass_rate # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."  # P95 < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model / bật cache |
| PII detected count | spike >10/giờ | Security alert |

---

## Kết quả thực tế từ Lab (offline / reproducible)

| Metric | Kết quả | Nguồn |
|---|---|---|
| RAGAS avg_score (50q) | 0.7948 | lexical-overlap proxy |
| Factual avg_score | 0.8445 | proxy |
| Multi-hop avg_score | 0.7615 | proxy |
| Adversarial avg_score | 0.7618 | proxy |
| Worst metric | context_precision (44/50) | proxy |
| Dominant failure distribution | factual | proxy |
| Cohen's κ (Judge vs Human) | 0.286 (fair) | heuristic judge |
| Position bias rate | 0% (judge tất định) | heuristic judge |
| Verbosity bias (thô) | 62.5% (artifact tập probe) | heuristic judge |
| Adversarial pass rate | **20/20 (100%)** | regex PII + keyword rail |
| Guard P95 latency (offline) | 0.12ms | regex/keyword path |

**Bonus đạt được:** Phase A (adversarial 0.7618 < factual 0.8445 ✓), Phase C (20/20 ≥ 18/20 ✓).
**Bonus chưa đạt offline:** Phase B (κ 0.286 < 0.6) — cần LLM judge thật.

---

## Nhận xét & Cải tiến

1. **RAGAS Quality:** Factual mạnh nhất (0.8445); adversarial thấp hơn (0.7618) do
   retrieve cả policy cũ lẫn mới → cần metadata filter ưu tiên version hiện hành.
   Điểm yếu chính là **context_precision** (44/50 câu) — top-k rộng kéo theo chunk
   nhiễu; siết reranking sẽ cải thiện nhiều nhất.

2. **LLM Judge:** Judge heuristic offline cho κ = 0.286 (fair) — bắt được dạng câu
   né tránh nhưng bỏ lỡ lỗi factual tinh vi (q5/q29/q41). Để đạt substantial (κ>0.6)
   cần GPT-4o-mini đối chiếu nội dung chính sách. Position bias = 0 (tất định), nên
   swap-and-average chủ yếu có giá trị khi nâng cấp lên judge LLM.

3. **Guardrails:** 100% adversarial block (4 PII qua regex + 16 jailbreak/off-topic/
   injection qua keyword rail). Latency offline cực thấp nhưng không đại diện; cần
   đo lại với Presidio + NeMo thật.

4. **Production improvements:** (a) semantic matching cho rail thay vì keyword,
   (b) cài Presidio + spaCy và thêm recognizer PII tiếng Việt (passport, STK ngân hàng),
   (c) circuit breaker khi NeMo latency tăng đột biến, (d) nâng judge lên LLM thật cho κ.
