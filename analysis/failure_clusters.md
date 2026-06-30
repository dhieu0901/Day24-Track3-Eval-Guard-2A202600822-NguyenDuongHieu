# Failure Cluster Analysis

**Sinh viên:** Nguyễn Dương Hiếu
**Ngày:** 30/06/2026
**Dataset:** 50 câu hỏi (20 factual + 20 multi-hop + 10 adversarial)

> **Chế độ chạy:** offline. Điểm RAGAS ở đây được tính bằng **bộ chấm điểm proxy
> dựa trên lexical-overlap** (`src/m4_eval.py::evaluate_ragas` fallback) vì package
> `ragas` + LLM API không chạy trong môi trường này. Số liệu mang tính **tương đối,
> tái lập được** từ `answers_50q.json`, **không phải** RAGAS thật chạy qua LLM.
> Tái tạo: `MOCK_EVAL=true python src/phase_a_ragas.py`.

---

## 1. Điểm trung bình theo distribution

| Distribution | n | avg_score | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|---|
| factual | 20 | **0.8445** | 0.9023 | 0.8507 | 0.7049 | 0.9202 |
| multi_hop | 20 | **0.7615** | 0.8205 | 0.7608 | 0.6438 | 0.8209 |
| adversarial | 10 | **0.7618** | 0.8797 | 0.7931 | 0.6168 | 0.7575 |
| **Tổng (50q)** | 50 | **0.7948** | — | — | — | — |

➡️ **Bonus Phase A đạt:** adversarial avg (0.7618) < factual avg (0.8445) — pipeline
phân biệt được câu hỏi bẫy version-conflict so với câu factual đơn giản.

---

## 2. Failure Matrix (worst_metric × distribution)

| Metric \ Distribution | Factual | Multi-hop | Adversarial | **Total** |
|---|---|---|---|---|
| faithfulness | 0 | 2 | 1 | 3 |
| answer_relevancy | 3 | 0 | 0 | 3 |
| context_precision | 17 | 18 | 9 | **44** |
| context_recall | 0 | 0 | 0 | 0 |

**Dominant failure distribution:** `factual`
**Dominant failure metric:** `context_precision`

---

## 3. Key Findings

1. **`context_precision` là điểm yếu áp đảo** — 44/50 câu có worst-metric là
   context_precision. Nguyên nhân: pipeline retrieve top-k khá rộng nên kéo theo
   nhiều chunk ít liên quan (irrelevant chunks), kéo precision xuống dù recall cao
   (context_recall trung bình rất tốt: 0.92 ở factual).
   - **Giải pháp:** thêm metadata filter + tăng chất lượng reranking để cắt bớt
     chunk nhiễu trước khi đưa vào LLM.

2. **Adversarial (0.7618) thấp hơn factual (0.8445)** — đúng kỳ vọng. Các câu này
   cố tình hỏi theo policy cũ (v2023) hoặc dùng phủ định/bẫy; pipeline retrieve cả
   phiên bản cũ lẫn mới nên context_precision tụt (0.6168 — thấp nhất trong 3 nhóm).

3. **Multi-hop (0.7615)** thấp tương đương adversarial, chủ yếu do faithfulness
   (2 câu) và context_precision (18 câu): tổng hợp nhiều tài liệu → dễ lẫn chunk
   không liên quan và dễ "chế" số liệu khi phải tính toán.

4. **Factual** mạnh nhất nhưng vẫn có 3 câu yếu ở `answer_relevancy` — câu trả lời
   lệch trọng tâm câu hỏi (vd: hỏi nghỉ khi vợ sinh, hỏi ai phê duyệt nghỉ không lương).

---

## 4. Bottom 10 Questions

| Rank | ID | Distribution | Avg Score | Worst Metric | Question |
|---|---|---|---|---|---|
| 1 | 30 | multi_hop | 0.5010 | faithfulness | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và chính thức |
| 2 | 33 | multi_hop | 0.5170 | faithfulness | Manager thâm niên 12 năm: tổng phụ cấp + ngày phép theo v2024 |
| 3 | 50 | adversarial | 0.5201 | faithfulness | Manager dùng VPN cá nhân (NordVPN) khi WFH được không? |
| 4 | 9 | factual | 0.6359 | answer_relevancy | Nam nhân viên nghỉ bao nhiêu ngày khi vợ sinh con? |
| 5 | 22 | multi_hop | 0.7156 | context_precision | Mua laptop 30 triệu: ai phê duyệt và cần gì từ CNTT? |
| 6 | 7 | factual | 0.7207 | answer_relevancy | Nghỉ phép không lương 20 ngày cần ai phê duyệt? |
| 7 | 34 | multi_hop | 0.7239 | context_precision | Tạm ứng 4 triệu vs 7 triệu: quy trình phê duyệt khác nhau? |
| 8 | 21 | multi_hop | 0.7246 | context_precision | Senior 9 năm thâm niên: số ngày phép năm và khoảng lương? |
| 9 | 45 | adversarial | 0.7383 | context_precision | Có cần kích hoạt MFA không? |
| 10 | 44 | adversarial | 0.7516 | context_precision | Bao lâu phải đổi mật khẩu một lần? |

---

## 5. Recommendations

| Priority | Action | Expected Impact |
|---|---|---|
| P0 | Metadata filter ưu tiên policy version mới nhất (v2024 > v2023) | Tăng context_precision cho adversarial/factual |
| P1 | Siết top-k + cải thiện reranker tiếng Việt để loại chunk nhiễu | Giảm 44 case worst=context_precision |
| P2 | Tinh chỉnh prompt cho câu factual lệch trọng tâm (answer_relevancy) | Cứu 3 case factual |
| P3 | Tăng cường chunking/multi-hop retrieval cho câu tính toán | Giảm faithfulness failure ở multi_hop |
