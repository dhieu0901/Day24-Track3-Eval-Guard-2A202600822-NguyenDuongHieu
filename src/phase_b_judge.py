from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


def _use_offline_judge() -> bool:
    """Offline khi: không có API key, MOCK_EVAL=true, hoặc đang chạy pytest.

    Giữ đường gọi LLM thật khi có key và không ở chế độ mock/test.
    """
    return (not OPENAI_API_KEY
            or os.environ.get("MOCK_EVAL") == "true"
            or "pytest" in sys.modules)


def _offline_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    """Judge xác định (deterministic) không cần API — dùng cho test/offline.

    Heuristic chấm điểm mỗi answer theo: (1) độ phủ từ khoá câu hỏi,
    (2) tính cụ thể (có số liệu/đơn vị), (3) độ súc tích. Hoàn toàn tái lập được.
    """
    q_tokens = set(re.findall(r"[0-9a-zà-ỹ]+", question.lower()))

    def score(ans: str) -> float:
        a_tokens = set(re.findall(r"[0-9a-zà-ỹ]+", ans.lower()))
        coverage = len(a_tokens & q_tokens) / (len(q_tokens) or 1)
        specificity = 1.0 if re.search(r"\d", ans) else 0.0
        conciseness = 1.0 - min(len(ans), 600) / 600 * 0.3
        return round(0.40 + 0.40 * coverage + 0.15 * specificity + 0.05 * conciseness, 3)

    sa, sb = min(score(answer_a), 1.0), min(score(answer_b), 1.0)
    if abs(sa - sb) < 0.02:
        winner = "tie"
    else:
        winner = "A" if sa > sb else "B"
    reasoning = (f"[offline heuristic] điểm A={sa}, B={sb} dựa trên độ phủ câu hỏi, "
                 f"tính cụ thể (số liệu) và độ súc tích.")
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": sa, "B": sb}}


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác (accuracy), đầy đủ (completeness), súc tích (conciseness).
Trả lời JSON (chỉ trả về định dạng JSON, không kèm giải thích hay ký tự markdown khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn lý do chọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''
    if _use_offline_judge():
        return _offline_pairwise(question, answer_a, answer_b)

    from openai import OpenAI
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user",   "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(resp.choices[0].message.content)
        
        # Normalize and validate outputs
        winner = result.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
            
        scores = result.get("scores", {"A": 0.0, "B": 0.0})
        scores["A"] = float(scores.get("A", 0.0))
        scores["B"] = float(scores.get("B", 0.0))
        
        return {
            "winner": winner,
            "reasoning": result.get("reasoning", ""),
            "scores": scores
        }
    except Exception as e:
        print(f"  [Error] LLM Judge call failed: {e}")
        return {"winner": "tie", "reasoning": f"Lỗi gọi API: {e}", "scores": {"A": 0.0, "B": 0.0}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Average: consensus only if both agree
    if pass1.get("winner") == winner_pass2:
        final = pass1.get("winner", "tie")
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1.get("winner") == winner_pass2)

    scores_1 = pass1.get("scores", {"A": 0.0, "B": 0.0})
    scores_2_raw = pass2_raw.get("scores", {"A": 0.0, "B": 0.0})
    scores_2 = {"A": scores_2_raw.get("B", 0.0), "B": scores_2_raw.get("A", 0.0)}

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1.get("winner", "tie"),
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=scores_1,
        scores_pass2=scores_2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n

    p_judge_1 = sum(1 for x in judge_labels if x == 1) / n
    p_human_1 = sum(1 for x in human_labels if x == 1) / n
    
    p_judge_0 = sum(1 for x in judge_labels if x == 0) / n
    p_human_0 = sum(1 for x in human_labels if x == 0) / n

    p_e = (p_judge_1 * p_human_1) + (p_judge_0 * p_human_0)

    if abs(1.0 - p_e) < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0

    κ = (p_o - p_e) / (1.0 - p_e)
    return κ


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0
            },
            "interpretation": "Không có dữ liệu đánh giá."
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive
        },
        "interpretation": interpretation,
    }


# ─── Pointwise quality label (offline) — phục vụ Cohen's κ ─────────────────────

_HEDGE_WORDS = ("miễn là", "có thể", "tùy", "vui lòng liên hệ", "liên hệ phòng",
                "không tìm thấy", "không rõ", "tham khảo thêm")


def offline_quality_label(question: str, answer: str) -> int:
    """Gán nhãn chất lượng 0/1 cho một câu trả lời (deterministic, không cần API).

    Heuristic: 'good' (1) nếu câu trả lời cụ thể (có số liệu), bám câu hỏi và
    không né tránh; 'bad' (0) nếu mơ hồ/né tránh. Đây là proxy offline — không
    bắt được lỗi factual tinh vi như LLM/human, nên κ sẽ ở mức vừa phải.
    """
    a_low = answer.lower()
    q_tokens = set(re.findall(r"[0-9a-zà-ỹ]+", question.lower()))
    a_tokens = set(re.findall(r"[0-9a-zà-ỹ]+", a_low))
    coverage = len(q_tokens & a_tokens)
    score = 0.0
    score += 0.5 if re.search(r"\d", answer) else 0.0      # có số liệu cụ thể
    score += 0.3 if coverage >= 2 else 0.0                 # bám câu hỏi
    score += 0.2 if len(answer) > 25 else 0.0              # đủ chi tiết
    score -= 0.4 if any(w in a_low for w in _HEDGE_WORDS) else 0.0  # né tránh
    return 1 if score >= 0.5 else 0


# Bộ cặp answer được biên soạn (better vs worse) để đo position/verbosity bias.
# Mỗi cặp: (question, answer_tốt_hơn, answer_kém_hơn). Cố ý cho độ dài đảo chiều
# để kiểm tra xem judge có thiên vị câu dài hơn không.
_BIAS_PROBE_PAIRS = [
    ("Nhân viên được nghỉ bao nhiêu ngày phép năm?",
     "Theo chính sách v2024 hiện hành, nhân viên được nghỉ 15 ngày phép năm.",
     "Nhân viên có 12 ngày phép."),
    ("Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?",
     "Nghỉ 3 ngày làm việc có lương khi kết hôn.",
     "Khi kết hôn, theo quy định chung của công ty thì nhân viên sẽ được sắp xếp một số ngày nghỉ phù hợp tùy trường hợp cụ thể."),
    ("Mua thiết bị 55 triệu cần ai phê duyệt?",
     "Vì 55 triệu vượt ngưỡng 50 triệu nên cần CEO phê duyệt.",
     "Cần Giám đốc phòng ban phê duyệt khoản chi này."),
    ("Thưởng Tết tối thiểu cho nhân viên chính thức ≥6 tháng là bao nhiêu?",
     "Tối thiểu 1 tháng lương.",
     "Mức thưởng Tết sẽ được công ty xem xét và quyết định dựa trên nhiều yếu tố khác nhau trong từng năm tài chính."),
    ("Nhân viên thử việc có được nghỉ phép năm không?",
     "Không; nhân viên thử việc chưa được hưởng phép năm, nếu cần phải xin nghỉ không lương.",
     "Có thể được nếu sắp xếp hợp lý."),
    ("Nhân viên có được dùng VPN cá nhân khi WFH không?",
     "Không; VPN cá nhân bị cấm, phải dùng VPN WireGuard của công ty theo chính sách v1.3.",
     "Được, miễn là đảm bảo kết nối an toàn."),
    ("Mật khẩu phải có tối thiểu bao nhiêu ký tự?",
     "Tối thiểu 12 ký tự theo chính sách mật khẩu v2.",
     "Mật khẩu cần đủ mạnh và nên thay đổi định kỳ để đảm bảo an toàn cho hệ thống."),
    ("Khóa học tài trợ 25 triệu, nghỉ sau 8 tháng phải hoàn trả bao nhiêu?",
     "Hoàn trả 100% (25 triệu) vì nghỉ trước mốc cam kết 12 tháng.",
     "Phải hoàn lại một phần chi phí tùy theo thỏa thuận."),
]


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = "offline heuristic" if _use_offline_judge() else f"live ({JUDGE_MODEL})"
    print(f"Phase B — LLM-as-Judge | mode: {mode}")

    # --- Task 5 + 6: pairwise + swap-and-average trên bộ cặp probe ---
    print(f"\nRunning swap-and-average on {len(_BIAS_PROBE_PAIRS)} pairs...")
    judge_results = []
    for q, better, worse in _BIAS_PROBE_PAIRS:
        jr = swap_and_average(q, better, worse)   # A = better, B = worse
        judge_results.append(jr)
        print(f"  [{jr.final_winner:>3}] consistent={jr.position_consistent} | {q[:45]}")

    # --- Task 7: Cohen's κ — judge (offline labels) vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    judge_labels = [offline_quality_label(it["question"], it["model_answer"]) for it in human_data]
    human_labels = [it["human_label"] for it in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\nCohen's κ (judge vs human, n={len(human_labels)}): {kappa:.3f}")

    # --- Task 8: bias report ---
    bias = bias_report(judge_results)
    print(f"Bias report: position_bias_rate={bias['position_bias_rate']} "
          f"verbosity_bias={bias['verbosity_bias']}")

    # Save reports/judge_results.json
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "results": [
                {
                    "question": r.question,
                    "answer_a": r.answer_a,
                    "answer_b": r.answer_b,
                    "winner_pass1": r.winner_pass1,
                    "winner_pass2": r.winner_pass2,
                    "final_winner": r.final_winner,
                    "position_consistent": r.position_consistent,
                    "scores_pass1": r.scores_pass1,
                    "scores_pass2": r.scores_pass2,
                }
                for r in judge_results
            ],
            "bias_report": bias,
            "cohen_kappa": {
                "judge_labels": judge_labels,
                "human_labels": human_labels,
                "kappa": round(kappa, 4),
            },
        }, f, indent=2, ensure_ascii=False)
    print("Saved reports/judge_results.json")
