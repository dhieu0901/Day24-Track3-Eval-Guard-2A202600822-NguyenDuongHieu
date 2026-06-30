# Bias Report: LLM-as-Judge Analysis

**Sinh viên:** Nguyễn Dương Hiếu
**Ngày:** 30/06/2026
**Methodology:** Swap-and-Average (2-pass pairwise) + Cohen's κ vs human labels

> **Chế độ chạy:** offline. Judge dùng **bộ chấm điểm heuristic tất định**
> (`src/phase_b_judge.py::_offline_pairwise` / `offline_quality_label`) thay cho
> gọi LLM thật, vì muốn kết quả **tái lập 100%** và không phụ thuộc API. Vì vậy
> các con số dưới đây phản ánh hành vi của *judge heuristic*, không phải GPT-4o-mini.
> Tái tạo: `MOCK_EVAL=true python src/phase_b_judge.py`.
> Muốn κ "thật" của LLM judge: bỏ `MOCK_EVAL` và đặt `OPENAI_API_KEY` rồi chạy lại.

---

## 1. Position Bias

| Metric | Giá trị |
|---|---|
| Tổng cặp đánh giá (probe pairs) | 8 |
| Position consistent (cả 2 pass đồng ý) | 8 |
| Position inconsistent | 0 |
| **Position bias rate** | **0.0%** |

**Đánh giá:** Judge heuristic **không có position bias** — đây là kết quả *kỳ vọng*:
hàm chấm điểm là tất định và đối xứng theo nội dung, nên đổi thứ tự A/B rồi quy đổi
lại luôn cho cùng winner. Đây cũng là lý do swap-and-average hữu ích: nó phơi bày
position bias ở judge LLM thật (vốn không tất định), còn ở judge heuristic thì
xác nhận bias = 0.

---

## 2. Verbosity Bias

| Metric | Giá trị |
|---|---|
| Decisive results (A hoặc B thắng) | 8 |
| Winner thắng VÀ dài hơn đối thủ | 5 |
| **Verbosity bias rate (thô)** | **62.5%** |

**Đánh giá — cần đọc thận trọng:** Trong 8 cặp probe, câu trả lời *đúng hơn* (A)
thắng cả 8/8; trong đó 5/8 trường hợp A tình cờ cũng dài hơn B. Chỉ số 62.5% vì vậy
**không** phải bằng chứng judge thiên vị câu dài — nó chỉ phản ánh việc câu đúng
trong bộ probe thường dài hơn. Trên thực tế hàm chấm điểm còn **trừ nhẹ điểm theo
độ dài** (conciseness term), nên judge heuristic không bị verbosity bias.

**Mitigation (cho judge LLM thật):**
- Thêm rubric "unnecessary verbosity penalty" vào prompt.
- Cân bằng độ dài A/B trong tập probe để tách bạch "đúng" khỏi "dài".

---

## 3. Cohen's Kappa (Judge vs Human)

| Metric | Giá trị |
|---|---|
| Judge labels | [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] |
| Human labels | [1, 0, 1, 1, 1, 0, 1, 0, 1, 0] |
| Observed agreement | 7/10 |
| **Cohen's κ** | **0.286** |

**Interpretation (Landis & Koch):**
- 0.00–0.20: Slight
- **0.21–0.40: Fair agreement** ← (0.286)
- 0.41–0.60: Moderate
- 0.61–0.80: Substantial
- 0.81–1.00: Almost perfect

**Đánh giá:** κ = 0.286 (**fair**). Judge heuristic đồng ý với human ở các câu mơ hồ/
né tránh (vd q50 "miễn là…", human cũng chấm 0), nhưng **bỏ lỡ 3 lỗi factual tinh vi**
mà heuristic không thể phát hiện:
- **q5:** "Giám đốc phòng ban phê duyệt" — sai vì 55 triệu > 50 triệu phải CEO duyệt.
- **q29:** thiếu vai trò Kế toán trưởng + tính sai phí phạt pro-rata.
- **q41:** trả lời 12 ngày theo v2023 đã hết hiệu lực (đúng phải 15 ngày v2024).

Cả 3 câu này đều "nhìn có vẻ tốt" (có số liệu, bám câu hỏi) nên heuristic gán 1,
trong khi human gán 0. Đây chính là **giới hạn của judge không-LLM**: bắt được dạng
né tránh nhưng không bắt được sai về *nội dung chính sách*.

**Để đạt bonus κ > 0.6 (substantial):** cần judge LLM thật (GPT-4o-mini) có khả năng
đối chiếu nội dung với ground-truth — chạy lại Phase B ở chế độ live.

---

## 4. Tổng kết

| Bias Type | Kết quả (offline heuristic) | Hành động |
|---|---|---|
| Position bias | 0% (judge tất định) | Swap-and-average vẫn cần cho judge LLM thật |
| Verbosity bias | 62.5% (artifact của tập probe, không phải bias thật) | Cân bằng độ dài tập probe |
| Judge-Human agreement | κ = 0.286 (fair) | Nâng cấp lên LLM judge để đạt substantial |
