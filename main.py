"""
ProjectHub AI Microservice
--------------------------
Runs independently on port 8001.
Node.js calls this service via HTTP — no external API, fully offline.

Features:
  1.  /plagiarism           — TF-IDF + Cosine Similarity
  2.  /summarize            — Extractive summarization (TF-IDF sentence scoring)
  3.  /chatbot              — Keyword + context NLP chatbot
  4.  /risk-predict         — Weighted ML scoring risk predictor
  5.  /smart-grade          — Academic rubric grader (Flesch + TTR + Vocab)
  6.  /viva-questions       — Keyword extraction + question templates
  7.  /milestone-risk       — Milestone completion risk analyzer
  8.  /predict-evaluation   — 4-category weighted score predictor
  9.  /generate-eval-report — Full text evaluation report generator

Run: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re
import math
from collections import Counter
import datetime

app = FastAPI(title="ProjectHub AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════

STOPWORDS = set([
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "this","that","these","those","it","its","we","our","they","their","as",
    "not","so","if","then","than","also","into","about","which","who","what",
    "how","when","where","all","each","any","i","you","he","she","we","us",
    "him","her","them","my","your","his","her","its","our","their","me"
])

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return [w for w in clean_text(text).split() if w not in STOPWORDS and len(w) > 2]

def get_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ══════════════════════════════════════════════════════════
# FEATURE 1: PLAGIARISM CHECKER — TF-IDF + Cosine Similarity
# ══════════════════════════════════════════════════════════

class PlagiarismRequest(BaseModel):
    target_title: str
    target_description: str
    other_projects: List[dict]   # [{id, title, description, studentName}]

def compute_tfidf_vector(tokens: List[str], idf: dict) -> dict:
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: (count / total) * idf.get(word, 0) for word, count in tf.items()}

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in common)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def build_idf(all_token_lists: List[List[str]]) -> dict:
    N = len(all_token_lists)
    idf = {}
    all_words = set(w for tokens in all_token_lists for w in tokens)
    for word in all_words:
        doc_count = sum(1 for tokens in all_token_lists if word in tokens)
        idf[word] = math.log((N + 1) / (doc_count + 1)) + 1
    return idf

@app.post("/plagiarism")
def check_plagiarism(req: PlagiarismRequest):
    target_text = req.target_title + " " + req.target_description
    target_tokens = tokenize(target_text)

    all_token_lists = [target_tokens]
    other_tokens_list = []
    for proj in req.other_projects:
        tokens = tokenize(proj.get("title","") + " " + proj.get("description",""))
        other_tokens_list.append(tokens)
        all_token_lists.append(tokens)

    idf = build_idf(all_token_lists)
    target_vec = compute_tfidf_vector(target_tokens, idf)

    comparisons = []
    for i, proj in enumerate(req.other_projects):
        other_vec = compute_tfidf_vector(other_tokens_list[i], idf)
        sim = cosine_similarity(target_vec, other_vec)
        score = round(sim * 100)

        # Find common significant words for reason
        common_words = [w for w in set(target_tokens) & set(other_tokens_list[i])
                        if idf.get(w, 0) > 1.0][:5]

        if score >= 60:
            level = "High"
            reason = f"High textual overlap detected. Shared key terms: {', '.join(common_words) if common_words else 'similar phrasing and concepts'}."
        elif score >= 30:
            level = "Medium"
            reason = f"Moderate similarity found. Common concepts: {', '.join(common_words) if common_words else 'some shared terminology'}."
        else:
            level = "Low"
            reason = "Low similarity. Projects appear sufficiently different in content and approach."

        comparisons.append({
            "projectId": proj.get("id", ""),
            "projectTitle": proj.get("title", ""),
            "studentName": proj.get("studentName", "Unknown"),
            "similarityScore": score,
            "similarityLevel": level,
            "reason": reason,
        })

    comparisons.sort(key=lambda x: x["similarityScore"], reverse=True)

    max_score = comparisons[0]["similarityScore"] if comparisons else 0
    if max_score >= 60:
        overall_risk = "High"
        summary = f"High similarity detected with {sum(1 for c in comparisons if c['similarityLevel'] == 'High')} project(s). Immediate review recommended."
    elif max_score >= 30:
        overall_risk = "Medium"
        summary = "Moderate similarity found with some projects. Review recommended to ensure originality."
    else:
        overall_risk = "Low"
        summary = "Project appears sufficiently original. No significant similarity detected."

    return {
        "success": True,
        "overallRisk": overall_risk,
        "summary": summary,
        "comparisons": comparisons,
        "algorithm": "TF-IDF + Cosine Similarity"
    }


# ══════════════════════════════════════════════════════════
# FEATURE 2: REPORT SUMMARIZER — Extractive Summarization
# ══════════════════════════════════════════════════════════

class SummarizeRequest(BaseModel):
    text: str
    num_sentences: Optional[int] = 5
    project_title: Optional[str] = ""

def score_sentence(sentence: str, word_freq: dict, total_words: int) -> float:
    tokens = tokenize(sentence)
    if not tokens:
        return 0.0
    score = sum(word_freq.get(w, 0) for w in tokens) / len(tokens)
    # Boost sentences with numbers, percentages, or technical terms
    if re.search(r'\d+', sentence):
        score *= 1.2
    # Boost longer sentences (more informative)
    if len(tokens) > 10:
        score *= 1.1
    return score

@app.post("/summarize")
def summarize_report(req: SummarizeRequest):
    text = req.text.strip()
    if len(text) < 100:
        raise HTTPException(status_code=400, detail="Text too short to summarize. Please provide at least 100 characters.")

    sentences = get_sentences(text)
    if len(sentences) < 3:
        raise HTTPException(status_code=400, detail="Not enough sentences to summarize. Please provide more content.")

    # Build word frequency map
    all_tokens = tokenize(text)
    word_freq = Counter(all_tokens)
    max_freq = max(word_freq.values()) if word_freq else 1
    # Normalize frequencies
    word_freq = {w: freq / max_freq for w, freq in word_freq.items()}

    # Score each sentence
    scored = []
    for i, sent in enumerate(sentences):
        score = score_sentence(sent, word_freq, len(all_tokens))
        # Boost first and last sentences (usually important)
        if i == 0:
            score *= 1.3
        elif i == len(sentences) - 1:
            score *= 1.1
        scored.append((score, i, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = min(req.num_sentences, len(sentences))
    top_sentences = sorted(scored[:top_n], key=lambda x: x[1])  # restore original order

    summary = " ".join(s[2] for s in top_sentences)

    # Extract key topics (most frequent meaningful words)
    key_topics = [word for word, _ in word_freq.items()
                  if len(word) > 4 and word_freq[word] > 0.3][:8]

    # Word count stats
    original_words = len(text.split())
    summary_words = len(summary.split())
    compression = round((1 - summary_words / original_words) * 100) if original_words > 0 else 0

    return {
        "success": True,
        "summary": summary,
        "keyTopics": key_topics,
        "stats": {
            "originalWords": original_words,
            "summaryWords": summary_words,
            "compressionRate": f"{compression}%",
            "sentencesExtracted": top_n,
            "totalSentences": len(sentences),
        },
        "algorithm": "Extractive Summarization (TF-IDF Sentence Scoring)"
    }


# ══════════════════════════════════════════════════════════
# FEATURE 3: SMART CHATBOT — Intent Detection + Context NLP
# ══════════════════════════════════════════════════════════

class ChatbotRequest(BaseModel):
    message: str
    project: Optional[dict] = None     # project data from MongoDB
    deadlines: Optional[List[dict]] = []

INTENT_MAP = [
    {"intent": "greeting",    "keywords": ["hello","hi","hey","good morning","good evening","howdy","namaste","helo","hii"]},
    {"intent": "help",        "keywords": ["help","what can you do","assist","support","guide","how to use","features"]},
    {"intent": "deadline",    "keywords": ["deadline","due date","due","submit by","last date","submission","when","time left","days left","remaining"]},
    {"intent": "progress",    "keywords": ["progress","score","performance","how am i doing","status","result","marks","grade","risk"]},
    {"intent": "files",       "keywords": ["file","upload","document","report","attachment","pdf","submitted","uploaded"]},
    {"intent": "feedback",    "keywords": ["feedback","comment","review","remarks","supervisor said","what did","suggestion"]},
    {"intent": "supervisor",  "keywords": ["supervisor","teacher","guide","mentor","assigned to","who is my","contact"]},
    {"intent": "milestone",   "keywords": ["milestone","task","phase","stage","checkpoint","target","goal","objective"]},
    {"intent": "proposal",    "keywords": ["proposal","project title","description","abstract","idea","topic","approved","rejected","pending"]},
    {"intent": "tips",        "keywords": ["tip","advice","suggest","improve","how to improve","recommendation","best practice"]},
]

def detect_intent(message: str) -> str:
    lower = message.lower()
    for rule in INTENT_MAP:
        for kw in rule["keywords"]:
            if kw in lower:
                return rule["intent"]
    return "unknown"

def format_date(date_str: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z",""))
        return dt.strftime("%d %b %Y")
    except:
        return str(date_str)

def build_reply(intent: str, project: dict, deadlines: list) -> str:
    now = datetime.datetime.utcnow()

    if intent == "greeting":
        title = project.get("title", "your project") if project else "your project"
        return f"Hello! 👋 I'm your ProjectHub AI Assistant.\nI can help you with information about **{title}**.\n\nTry asking me:\n• What is my deadline?\n• What is my progress score?\n• How many files have I uploaded?\n• Do I have any feedback?"

    if intent == "help":
        return ("Here's what I can help you with:\n\n"
                "📅 **Deadlines** — 'When is my deadline?' or 'How many days left?'\n"
                "📊 **Progress** — 'What is my progress?' or 'Am I at risk?'\n"
                "📁 **Files** — 'How many files uploaded?' or 'What did I submit?'\n"
                "💬 **Feedback** — 'Do I have feedback?' or 'What did supervisor say?'\n"
                "👨‍🏫 **Supervisor** — 'Who is my supervisor?'\n"
                "🎯 **Milestones** — 'What are my milestones?'\n"
                "💡 **Tips** — 'Give me tips to improve my project'\n"
                "📝 **Proposal** — 'What is my project status?'")

    if not project:
        return "⚠️ You don't have a project yet. Please submit a proposal first from the **Submit Proposal** section."

    status = project.get("status", "pending")
    files = project.get("files", [])
    feedback = project.get("feedback", [])
    title = project.get("title", "your project")
    deadline_str = project.get("deadline")

    # Parse deadlines
    upcoming = []
    overdue = []
    for d in deadlines:
        try:
            due = datetime.datetime.fromisoformat(d.get("dueDate","").replace("Z",""))
            if due < now:
                overdue.append(d)
            else:
                upcoming.append(d)
        except:
            pass
    upcoming.sort(key=lambda x: x.get("dueDate",""))

    if intent == "deadline":
        parts = []
        if deadline_str:
            try:
                final_due = datetime.datetime.fromisoformat(deadline_str.replace("Z",""))
                days_left = (final_due - now).days
                if days_left > 0:
                    parts.append(f"📅 **Final Deadline:** {format_date(deadline_str)}\n⏳ **{days_left} days remaining**")
                else:
                    parts.append(f"📅 **Final Deadline:** {format_date(deadline_str)}\n🚨 **Overdue by {abs(days_left)} days!**")
            except:
                parts.append(f"📅 **Final Deadline:** {deadline_str}")
        if upcoming:
            parts.append(f"⏰ **Next Milestone:** \"{upcoming[0].get('name','?')}\" — due {format_date(upcoming[0].get('dueDate',''))}")
        if overdue:
            names = ", ".join(d.get('name','?') for d in overdue[:3])
            parts.append(f"⚠️ **Overdue Milestones ({len(overdue)}):** {names}")
        if not parts:
            return "📅 No deadlines have been set for your project yet. Contact your supervisor or admin."
        return "\n\n".join(parts)

    if intent == "progress":
        score = 0
        if status == "completed": score += 30
        elif status == "approved": score += 20
        elif status == "pending": score += 10
        score += min(len(files) * 5, 20)
        score += min(len(feedback) * 5, 20)
        if len(overdue) == 0 and len(deadlines) > 0: score += 30
        elif len(deadlines) > 0: score += max(0, 30 - len(overdue) * 10)
        else: score += 15
        score = min(score, 100)
        emoji = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
        risk = "Low" if score >= 70 else "Medium" if score >= 40 else "High"
        verdict = "Great work! Keep it up 🎉" if score >= 70 else "Moderate progress. Focus on pending tasks." if score >= 40 else "Needs immediate attention. Please talk to your supervisor."
        return (f"📊 **Progress Report for \"{title}\"**\n\n"
                f"**Score:** {score}/100\n"
                f"**Risk Level:** {emoji} {risk}\n"
                f"**Project Status:** {status.capitalize()}\n"
                f"**Files Uploaded:** {len(files)}\n"
                f"**Feedback Received:** {len(feedback)}\n"
                f"**Overdue Milestones:** {len(overdue)}\n\n"
                f"_{verdict}_")

    if intent == "files":
        if not files:
            return "📁 No files uploaded yet.\n\nGo to **Upload Files** section to submit your documents, reports, or source code."
        recent = files[-3:]
        file_list = "\n".join(f"• {f.get('originalName','file')} ({f.get('fileType','unknown')})" for f in recent)
        return f"📁 **Files Uploaded: {len(files)}**\n\nRecent uploads:\n{file_list}\n\nHead to the **Upload Files** section to view or add more."

    if intent == "feedback":
        if not feedback:
            return "💬 No feedback received yet.\n\nYour supervisor will provide feedback after reviewing your work. Keep uploading your progress files!"
        latest = feedback[-1]
        msg = latest.get("message","")
        preview = msg[:200] + "..." if len(msg) > 200 else msg
        return (f"💬 **Feedback Summary**\n\n"
                f"**Total Feedback:** {len(feedback)}\n"
                f"**Latest:** {latest.get('title','Feedback')}\n\n"
                f"_{preview}_\n\n"
                f"Visit the **Feedback** section to read all feedback in detail.")

    if intent == "supervisor":
        sup = project.get("supervisor")
        if not sup or (isinstance(sup, dict) and not sup.get("name")):
            return "👨‍🏫 No supervisor has been assigned yet.\n\nGo to the **Supervisor** section to send a request, or contact your admin."
        name = sup.get("name","your supervisor") if isinstance(sup, dict) else "your supervisor"
        email = sup.get("email","") if isinstance(sup, dict) else ""
        return (f"👨‍🏫 **Your Supervisor:** {name}\n"
                + (f"📧 **Email:** {email}\n" if email else "")
                + "\nVisit the **Supervisor** section for more details.")

    if intent == "milestone":
        total = len(deadlines)
        if total == 0:
            return "🎯 No milestones have been set yet.\n\nYour supervisor or admin will create milestones for your project soon."
        next_up = f"\n**Next:** \"{upcoming[0].get('name','?')}\" — {format_date(upcoming[0].get('dueDate',''))}" if upcoming else ""
        return (f"🎯 **Milestones for \"{title}\"**\n\n"
                f"**Total:** {total}\n"
                f"**Upcoming:** {len(upcoming)}\n"
                f"**Overdue:** {len(overdue)}"
                f"{next_up}")

    if intent == "proposal":
        colors = {"approved":"✅","pending":"⏳","rejected":"❌","completed":"🏆"}
        icon = colors.get(status, "📝")
        desc = project.get("description","")
        preview = desc[:150] + "..." if len(desc) > 150 else desc
        return (f"📝 **Project: \"{title}\"**\n\n"
                f"**Status:** {icon} {status.capitalize()}\n\n"
                f"**Description:**\n_{preview}_\n\n"
                + ("Your project is approved! Keep working hard. 🎉" if status == "approved"
                   else "Your project is pending review by your supervisor." if status == "pending"
                   else "Your project was rejected. Please contact your supervisor for feedback." if status == "rejected"
                   else "Project completed! Great work! 🏆"))

    if intent == "tips":
        tips = []
        if len(files) == 0: tips.append("📁 Upload your project files (report, code, presentation) to show progress.")
        if len(feedback) == 0: tips.append("💬 Request feedback from your supervisor regularly.")
        if len(overdue) > 0: tips.append(f"⚠️ Complete your {len(overdue)} overdue milestone(s) immediately.")
        if status == "pending": tips.append("📝 Follow up with your supervisor to get your proposal approved.")
        tips += [
            "📅 Review your milestones every week to stay on track.",
            "📖 Document your progress regularly for a strong final report.",
            "🤝 Maintain regular communication with your supervisor.",
        ]
        return "💡 **Tips to Improve Your Project:**\n\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(tips[:5]))

    # Unknown intent
    return ("🤔 I'm not sure I understood that. Try asking:\n\n"
            "• 'What is my deadline?'\n"
            "• 'What is my progress score?'\n"
            "• 'How many files did I upload?'\n"
            "• 'Do I have any feedback?'\n"
            "• 'Who is my supervisor?'\n"
            "• 'Give me tips to improve'")

@app.post("/chatbot")
def chatbot(req: ChatbotRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")

    intent = detect_intent(req.message)
    reply = build_reply(intent, req.project or {}, req.deadlines or [])

    return {
        "success": True,
        "reply": reply,
        "intent": intent,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


# ══════════════════════════════════════════════════════════
# FEATURE 4: RISK PREDICTOR — Weighted ML Scoring Model
# ══════════════════════════════════════════════════════════

class RiskRequest(BaseModel):
    projectStatus: str           # pending / approved / completed / rejected
    filesCount: int
    feedbackCount: int
    totalDeadlines: int
    overdueCount: int
    upcomingCount: int
    daysUntilDeadline: Optional[int] = None
    daysSinceLastUpload: Optional[int] = None
    # New milestone signals
    totalMilestones: Optional[int] = 0
    approvedMilestones: Optional[int] = 0
    submittedMilestones: Optional[int] = 0
    rejectedMilestones: Optional[int] = 0
    overdueMilestones: Optional[int] = 0
    milestoneCompletionRate: Optional[int] = 0

@app.post("/risk-predict")
def predict_risk(req: RiskRequest):
    score = 100  # Start from 100, deduct based on bad signals

    # 1. Project status weight
    status_penalty = {"pending": 20, "rejected": 40, "approved": 5, "completed": 0}
    score -= status_penalty.get(req.projectStatus, 15)

    # 2. Files uploaded (max benefit: 20 points)
    file_bonus = min(req.filesCount * 4, 20)
    score = score - 20 + file_bonus

    # 3. Overdue legacy deadlines (kept for backward compat, lighter weight now)
    score -= min(req.overdueCount * 6, 18)

    # 4. Feedback signals engagement
    feedback_bonus = min(req.feedbackCount * 3, 12)
    score += feedback_bonus

    # 5. Days until final deadline
    if req.daysUntilDeadline is not None:
        if req.daysUntilDeadline < 0:
            score -= 25
        elif req.daysUntilDeadline < 7:
            score -= 15
        elif req.daysUntilDeadline < 30:
            score -= 5

    # 6. Staleness (no uploads recently)
    if req.daysSinceLastUpload is not None:
        if req.daysSinceLastUpload > 30:
            score -= 10
        elif req.daysSinceLastUpload > 14:
            score -= 5

    # 7. Real Milestone signals (main new signals)
    total_ms = req.totalMilestones or 0
    if total_ms > 0:
        # Completion rate bonus/penalty  (max +15 / -20)
        cr = req.milestoneCompletionRate or 0
        if cr >= 80:
            score += 15
        elif cr >= 50:
            score += 5
        elif cr >= 25:
            score -= 10
        else:
            score -= 20

        # Overdue milestones heavy penalty
        score -= min((req.overdueMilestones or 0) * 10, 30)

        # Rejected milestones signal poor quality
        score -= min((req.rejectedMilestones or 0) * 5, 15)

        # Submitted milestones show engagement (pending supervisor approval)
        score += min((req.submittedMilestones or 0) * 3, 9)

    score = max(0, min(score, 100))

    # Determine risk level
    if score >= 65:
        risk_level = "Low"
        risk_color = "green"
        prediction = "On-track"
        description = "Your project is progressing well. Maintain your current pace."
        action = "Keep uploading files regularly and stay in touch with your supervisor."
    elif score >= 40:
        risk_level = "Medium"
        risk_color = "amber"
        prediction = "At-risk"
        description = "Some areas need attention. A few signals suggest the project may face challenges."
        action = "Focus on completing overdue milestones and request feedback from your supervisor."
    else:
        risk_level = "High"
        risk_color = "red"
        prediction = "Critical"
        description = "Project is in critical state. Immediate action is required."
        action = "Contact your supervisor immediately, complete overdue work, and upload all pending documents."

    # Feature importance (explains why this score)
    factors = []
    if req.overdueMilestones and req.overdueMilestones > 0:
        factors.append({"factor": f"{req.overdueMilestones} overdue weekly milestone(s)", "impact": "negative", "weight": "high"})
    if req.overdueCount > 0:
        factors.append({"factor": f"{req.overdueCount} overdue deadline(s)", "impact": "negative", "weight": "medium"})
    if req.filesCount == 0:
        factors.append({"factor": "No files uploaded", "impact": "negative", "weight": "medium"})
    elif req.filesCount >= 3:
        factors.append({"factor": f"{req.filesCount} files uploaded", "impact": "positive", "weight": "medium"})
    if req.feedbackCount > 0:
        factors.append({"factor": f"{req.feedbackCount} feedback(s) received", "impact": "positive", "weight": "low"})
    if req.daysUntilDeadline is not None and req.daysUntilDeadline < 14:
        factors.append({"factor": f"Only {req.daysUntilDeadline} days until deadline", "impact": "negative", "weight": "high"})
    if req.projectStatus == "approved":
        factors.append({"factor": "Project proposal approved", "impact": "positive", "weight": "medium"})
    total_ms = req.totalMilestones or 0
    if total_ms > 0:
        cr = req.milestoneCompletionRate or 0
        if cr >= 80:
            factors.append({"factor": f"{cr}% milestones completed", "impact": "positive", "weight": "high"})
        elif cr >= 50:
            factors.append({"factor": f"{cr}% milestones completed", "impact": "positive", "weight": "medium"})
        else:
            factors.append({"factor": f"Only {cr}% milestones completed", "impact": "negative", "weight": "high"})
        if (req.rejectedMilestones or 0) > 0:
            factors.append({"factor": f"{req.rejectedMilestones} milestone(s) rejected by supervisor", "impact": "negative", "weight": "medium"})
        if (req.submittedMilestones or 0) > 0:
            factors.append({"factor": f"{req.submittedMilestones} milestone(s) pending approval", "impact": "positive", "weight": "low"})

    return {
        "success": True,
        "riskScore": score,
        "riskLevel": risk_level,
        "riskColor": risk_color,
        "prediction": prediction,
        "description": description,
        "actionRequired": action,
        "factors": factors,
        "algorithm": "Weighted Feature Scoring Model"
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 5: SMART REPORT GRADER
# ── Pure Python, zero external libraries, zero APIs ──
#
# HOW TO ADD: Paste this entire block into your existing main.py file,
# just ABOVE the "# HEALTH CHECK" section at the bottom.
# ══════════════════════════════════════════════════════════════════════════════

ACADEMIC_VOCAB = set([
    "analysis","approach","assessment","assumption","authority","available",
    "benefit","concept","conclusion","consistent","context","contract","data",
    "definition","derived","distribution","economic","environment","established",
    "evaluation","evidence","factors","financial","formula","function",
    "identified","impact","implies","implementation","indicate","individual",
    "interpretation","involved","issues","methodology","obtained","participation",
    "percent","period","policy","potential","principle","procedure","process",
    "required","research","response","role","section","significant","similar",
    "source","specific","structure","system","theory","variables","framework",
    "hypothesis","literature","objective","paradigm","qualitative","quantitative",
    "sample","statistical","valid","algorithm","architecture","classification",
    "component","configuration","database","deployment","development",
    "documentation","efficiency","integration","interface","module","network",
    "optimization","parameter","performance","prototype","requirement",
    "scalability","security","testing","abstract","background","comparison",
    "complexity","constraint","correlation","dataset","description","design",
    "detection","discussion","experiment","feature","generation",
    "input","mechanism","model","output","pattern","prediction","proposed",
    "review","simulation","solution","technique","validation","workflow",
])

TRANSITION_WORDS = [
    "furthermore","however","therefore","consequently","additionally","moreover",
    "nevertheless","subsequently","accordingly","meanwhile","alternatively",
    "similarly","conversely","firstly","secondly","thirdly","finally",
    "in conclusion","in summary","to summarize","as a result","for instance",
    "for example","in contrast","on the other hand","in addition","in particular",
    "specifically","notably","significantly","thus","hence","whereby","nonetheless",
    "on the contrary","in comparison","to illustrate","as mentioned","overall",
]

SECTION_PATTERN = re.compile(
    r'\b(abstract|introduction|methodology|methods|results?|discussion|'
    r'conclusion|references?|background|literature\s+review|related\s+work|'
    r'system\s+design|implementation|evaluation|testing|future\s+work|'
    r'problem\s+statement|objectives?|scope|overview)\b',
    re.IGNORECASE,
)

def _count_syllables(word: str) -> int:
    word = word.lower().rstrip("es")
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    return max(1, count)

def _flesch(text: str) -> float:
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    if not words or not sentences:
        return 50.0
    asl = len(words) / len(sentences)
    asw = sum(_count_syllables(w) for w in words) / len(words)
    return max(0, min(100, round(206.835 - 1.015 * asl - 84.6 * asw, 1)))

def _grade_vocab(text: str) -> dict:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    total = max(len(words), 1)
    hits  = [w for w in words if w in ACADEMIC_VOCAB]
    density = len(hits) / total * 100
    unique  = len(set(hits))
    score   = min(25, round(density * 5))
    if score >= 20:   level = "Excellent"
    elif score >= 14: level = "Good"
    elif score >= 8:  level = "Satisfactory"
    else:             level = "Needs Work"
    return {"score": score, "maxScore": 25, "density": round(density, 2),
            "uniqueTerms": unique, "totalWords": total, "level": level,
            "comment": f"Found {unique} unique academic terms ({round(density,1)}% of text)."}

def _grade_structure(text: str) -> dict:
    found = list(set(m.group().lower() for m in SECTION_PATTERN.finditer(text)))
    score = min(20, len(found) * 4)
    if score >= 16:   level = "Excellent"
    elif score >= 10: level = "Good"
    elif score >= 6:  level = "Satisfactory"
    else:             level = "Needs Work"
    return {"score": score, "maxScore": 20, "sectionsFound": found,
            "sectionCount": len(found), "level": level,
            "comment": f"Detected {len(found)} section(s): {', '.join(found[:6]) or 'none'}."}

def _grade_coherence(text: str) -> dict:
    lower = text.lower()
    total_words = max(len(text.split()), 1)
    count = sum(lower.count(tw) for tw in TRANSITION_WORDS)
    density = count / total_words * 100
    score = min(20, round(density * 25))
    if score >= 16:   level = "Excellent"
    elif score >= 10: level = "Good"
    elif score >= 5:  level = "Satisfactory"
    else:             level = "Needs Work"
    return {"score": score, "maxScore": 20, "transitionCount": count,
            "density": round(density, 2), "level": level,
            "comment": f"{count} transition words found ({round(density,2)}% density)."}

def _grade_depth(text: str) -> dict:
    tokens = tokenize(text)
    total  = max(len(tokens), 1)
    unique = len(set(tokens))
    ttr    = unique / total
    word_count   = len(text.split())
    length_score = min(10, word_count // 40)
    depth_score  = min(10, round(ttr * 18))
    score = length_score + depth_score
    if score >= 16:   level = "Excellent"
    elif score >= 10: level = "Good"
    elif score >= 6:  level = "Satisfactory"
    else:             level = "Needs Work"
    return {"score": score, "maxScore": 20, "wordCount": word_count,
            "uniqueConcepts": unique, "typeTokenRatio": round(ttr, 3),
            "lengthScore": length_score, "depthScore": depth_score,
            "level": level,
            "comment": f"{word_count} words, {unique} unique concepts (TTR {round(ttr,3)})."}

def _grade_readability(text: str) -> dict:
    fre  = _flesch(text)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    asl  = len(words) / max(len(sents), 1)
    asw  = sum(_count_syllables(w) for w in words) / max(len(words), 1)
    fkgl = round(0.39 * asl + 11.8 * asw - 15.59, 1)
    if 30 <= fre <= 60:
        score, note = 15, "Ideal college-level readability."
    elif 20 <= fre < 30 or 60 < fre <= 70:
        score, note = 11, "Slightly outside ideal academic range."
    elif fre < 20:
        score, note = 7,  "Very complex — consider splitting long sentences."
    else:
        score, note = 9,  "Too simple for academic writing."
    if score >= 13:   level = "Excellent"
    elif score >= 9:  level = "Good"
    else:             level = "Needs Work"
    return {"score": score, "maxScore": 15,
            "fleschReadingEase": fre, "fleschKincaidGrade": fkgl,
            "avgSentenceLength": round(asl, 1), "avgSyllablesPerWord": round(asw, 2),
            "level": level, "comment": note}

def _sentence_stats(text: str) -> dict:
    sents  = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if not sents:
        return {"distribution": [], "avgLength": 0, "shortCount": 0, "longCount": 0, "totalSentences": 0}
    lengths = [len(s.split()) for s in sents]
    avg     = sum(lengths) / len(lengths)
    buckets = {"1-8": 0, "9-15": 0, "16-25": 0, "26-35": 0, "36+": 0}
    for ln in lengths:
        if   ln <= 8:  buckets["1-8"]   += 1
        elif ln <= 15: buckets["9-15"]  += 1
        elif ln <= 25: buckets["16-25"] += 1
        elif ln <= 35: buckets["26-35"] += 1
        else:          buckets["36+"]   += 1
    return {
        "distribution": [{"range": k, "count": v} for k, v in buckets.items()],
        "avgLength": round(avg, 1),
        "shortCount": buckets["1-8"],
        "longCount": buckets["36+"],
        "totalSentences": len(sents),
    }

class GradeRequest(BaseModel):
    text: str
    project_title: Optional[str] = ""
    student_name:  Optional[str] = ""

@app.post("/smart-grade")
def smart_grade(req: GradeRequest):
    text = req.text.strip()
    if len(text) < 100:
        raise HTTPException(400, "Text too short — provide at least 100 characters.")

    vocab       = _grade_vocab(text)
    structure   = _grade_structure(text)
    coherence   = _grade_coherence(text)
    depth       = _grade_depth(text)
    readability = _grade_readability(text)
    sent_stats  = _sentence_stats(text)

    total = vocab["score"] + structure["score"] + coherence["score"] + \
            depth["score"] + readability["score"]
    pct = round(total)

    if pct >= 85: grade, label, color = "A", "Distinction",     "green"
    elif pct >= 70: grade, label, color = "B", "Merit",         "blue"
    elif pct >= 55: grade, label, color = "C", "Pass",          "amber"
    elif pct >= 40: grade, label, color = "D", "Marginal Pass", "orange"
    else:           grade, label, color = "F", "Fail",          "red"

    radar = [
        {"subject": "Vocabulary",  "score": round(vocab["score"]       / 25  * 100), "fullMark": 100},
        {"subject": "Structure",   "score": round(structure["score"]   / 20  * 100), "fullMark": 100},
        {"subject": "Coherence",   "score": round(coherence["score"]   / 20  * 100), "fullMark": 100},
        {"subject": "Depth",       "score": round(depth["score"]       / 20  * 100), "fullMark": 100},
        {"subject": "Readability", "score": round(readability["score"] / 15  * 100), "fullMark": 100},
    ]

    strengths, improvements = [], []
    for name, res, threshold in [
        ("Academic vocabulary", vocab,       15),
        ("Document structure",  structure,   12),
        ("Coherence and flow",  coherence,   12),
        ("Content depth",       depth,       12),
        ("Readability",         readability,  9),
    ]:
        if res["score"] >= threshold:
            strengths.append(f"{name} — {res['level'].lower()}")
        else:
            improvements.append(res["comment"])

    return {
        "success": True,
        "totalScore": total, "percentage": pct,
        "grade": grade, "gradeLabel": label, "gradeColor": color,
        "projectTitle": req.project_title, "studentName": req.student_name,
        "criteria": {
            "academicVocabulary": vocab,
            "documentStructure":  structure,
            "coherenceFlow":      coherence,
            "contentDepth":       depth,
            "readability":        readability,
        },
        "radarData":    radar,
        "sentenceStats": sent_stats,
        "strengths":    strengths    or ["Keep developing all areas."],
        "improvements": improvements or ["Excellent — maintain this quality!"],
        "algorithm": "Rubric NLP Grader (Vocab + Structure + Coherence + TTR + Flesch)",
    }



# ══════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════

@app.get("/")
def health():
    return {"status": "ok", "service": "ProjectHub AI", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "features": [
        "plagiarism", "summarize", "chatbot", "risk-predict",
        "smart-grade", "viva-questions", "milestone-risk",
        "predict-evaluation", "generate-eval-report"
    ]}


# ══════════════════════════════════════════════════════════
# FEATURE 6: VIVA QUESTION GENERATOR
# ══════════════════════════════════════════════════════════

class VivaRequest(BaseModel):
    title: str
    description: str
    report_text: Optional[str] = ""
    status: Optional[str] = "approved"

QUESTION_TEMPLATES = {
    "Introduction": [
        "Can you briefly explain what your project {title} is about?",
        "What problem does your project {title} solve?",
        "What motivated you to work on {title}?",
        "Who are the intended users of your system?",
        "What are the main objectives of your project?",
        "How did you define the scope of your project?",
    ],
    "Technical": [
        "What technology stack did you use and why?",
        "What were the main technical challenges you faced?",
        "How does your system handle data storage?",
        "What is the architecture of your application?",
        "How did you ensure the security of your application?",
        "What algorithms or data structures did you use?",
        "How did you handle error cases in your system?",
        "What database did you use and why?",
        "How did you manage API communication in your project?",
        "What is the time and space complexity of your core algorithm?",
    ],
    "Design": [
        "Can you explain the overall system design of your project?",
        "How did you design the user interface?",
        "What design patterns did you apply?",
        "How did you ensure your system is scalable?",
        "What is the ER diagram of your database?",
        "How does data flow through your system?",
        "What are the main modules in your project?",
    ],
    "Testing": [
        "How did you test your application?",
        "What types of testing did you perform?",
        "Did you encounter any major bugs? How did you fix them?",
        "What tools did you use for testing?",
        "How do you validate user inputs?",
        "What would you do differently to improve reliability?",
    ],
    "Future Work": [
        "What limitations does your current system have?",
        "What features would you add if you had more time?",
        "How would you scale your system for 10x the current users?",
        "What improvements would you make to the UI/UX?",
        "How could machine learning be integrated into your project?",
        "What security enhancements would you add in the future?",
    ],
    "Reflection": [
        "What is the most important thing you learned from this project?",
        "How did working on this project improve your software engineering skills?",
        "If you had to redo the project, what would you do differently?",
        "How did you divide the work among your group members?",
        "What resources were most helpful during development?",
    ],
}

def extract_keywords_from_text(title: str, description: str, report: str) -> list:
    combined = title + " " + description + " " + report
    tokens = tokenize(combined)
    freq = Counter(tokens)
    return [w for w, c in freq.most_common(15) if len(w) > 4]

@app.post("/viva-questions")
def viva_questions(req: VivaRequest):
    keywords = extract_keywords_from_text(req.title, req.description, req.report_text or "")
    
    selected = {}
    counts = {"Introduction": 3, "Technical": 4, "Design": 3, "Testing": 2, "Future Work": 2, "Reflection": 2}
    
    for category, count in counts.items():
        templates = QUESTION_TEMPLATES.get(category, [])
        questions = []
        for t in templates[:count + 2]:
            q = t.replace("{title}", f'"{req.title}"')
            questions.append(q)
        selected[category] = questions[:count]
    
    # Add keyword-specific questions
    keyword_questions = []
    for kw in keywords[:5]:
        keyword_questions.append(f"You mentioned '{kw}' in your project. Can you explain it in detail?")
        keyword_questions.append(f"How does '{kw}' contribute to the core functionality of your system?")
    
    flat = []
    categories_out = []
    for cat, qs in selected.items():
        categories_out.append({"category": cat, "count": len(qs)})
        for q in qs:
            flat.append({"category": cat, "question": q, "difficulty": "medium"})
    
    if keyword_questions:
        flat.append({"category": "Project-Specific", "question": keyword_questions[0], "difficulty": "hard"})
        if len(keyword_questions) > 1:
            flat.append({"category": "Project-Specific", "question": keyword_questions[2] if len(keyword_questions)>2 else keyword_questions[1], "difficulty": "hard"})
        categories_out.append({"category": "Project-Specific", "count": min(2, len(keyword_questions))})

    tips = [
        "Understand every line of your code — examiners may ask about specific implementation details.",
        "Be able to explain your system design with a diagram from memory.",
        "Know the limitations of your project and have honest answers ready.",
        "Practice explaining your project in 2 minutes to a non-technical person.",
        "Revise the theory behind every technology you used.",
        "Be confident — you built this system, you know it best.",
    ]

    return {
        "success": True,
        "questions": flat,
        "categories": categories_out,
        "tips": tips,
        "totalQuestions": len(flat),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 8: MILESTONE RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class MilestoneRiskRequest(BaseModel):
    total: int
    approved: int
    submitted: int
    rejected: int
    overdue: int
    active: int
    upcoming: int
    completionRate: int
    withLogs: int
    projectStatus: str

@app.post("/milestone-risk")
def milestone_risk(req: MilestoneRiskRequest):
    score = 100

    # No milestones at all
    if req.total == 0:
        return {
            "riskScore": 50,
            "riskLevel": "Medium",
            "prediction": "No Data",
            "description": "No weekly milestones have been created for this project yet.",
            "actionRequired": "Create weekly milestones to track project progress properly.",
            "factors": [{"factor": "No milestones created", "impact": "negative", "weight": "high"}],
        }

    # Completion rate (most important signal)
    cr = req.completionRate
    if cr >= 80:   score += 20
    elif cr >= 60: score += 10
    elif cr >= 40: score -= 10
    elif cr >= 20: score -= 20
    else:          score -= 35

    # Overdue penalty (heavy)
    score -= min(req.overdue * 12, 36)

    # Rejected milestones — quality signal
    score -= min(req.rejected * 8, 24)

    # Log entries show student is actually working
    log_ratio = req.withLogs / req.total if req.total > 0 else 0
    if log_ratio >= 0.8:   score += 15
    elif log_ratio >= 0.5: score += 8
    elif log_ratio >= 0.2: score += 2
    else:                  score -= 10

    # Submitted waiting for approval — good signal
    score += min(req.submitted * 4, 12)

    # Project status
    if req.projectStatus == "completed": score += 10
    elif req.projectStatus == "rejected": score -= 20

    score = max(0, min(score, 100))

    if score >= 70:
        level, prediction = "Low", "On-track"
        description = "Milestone progress is healthy. Students are consistently logging work and getting approvals."
        action = "Continue the current pace. Ensure upcoming milestones are submitted on time."
    elif score >= 45:
        level, prediction = "Medium", "At-risk"
        description = "Some milestone delays detected. Intervention may be needed to get back on track."
        action = "Review overdue milestones with the student and set a catch-up schedule."
    else:
        level, prediction = "High", "Critical"
        description = "Milestone progress is critically behind. Significant delays and missing submissions detected."
        action = "Schedule an immediate meeting. Overdue milestones must be addressed urgently."

    factors = []
    if req.overdue > 0:
        factors.append({"factor": f"{req.overdue} overdue milestone(s)", "impact": "negative", "weight": "high"})
    if req.rejected > 0:
        factors.append({"factor": f"{req.rejected} milestone(s) rejected by supervisor", "impact": "negative", "weight": "medium"})
    if req.approved > 0:
        factors.append({"factor": f"{req.approved} milestone(s) approved", "impact": "positive", "weight": "high"})
    if log_ratio >= 0.5:
        factors.append({"factor": f"{req.withLogs}/{req.total} milestones have work logs", "impact": "positive", "weight": "medium"})
    elif log_ratio < 0.3:
        factors.append({"factor": "Most milestones missing work log entries", "impact": "negative", "weight": "medium"})
    if req.submitted > 0:
        factors.append({"factor": f"{req.submitted} milestone(s) submitted, awaiting approval", "impact": "positive", "weight": "low"})

    return {
        "riskScore": score,
        "riskLevel": level,
        "prediction": prediction,
        "description": description,
        "actionRequired": action,
        "factors": factors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 9: EVALUATION SCORE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class EvalPredictRequest(BaseModel):
    projectStatus: str
    filesCount: int
    feedbackCount: int
    milestoneStats: dict
    existingScores: Optional[dict] = None
    existingTotal: Optional[float] = None
    isFinalized: bool = False

@app.post("/predict-evaluation")
def predict_evaluation(req: EvalPredictRequest):
    ms = req.milestoneStats
    cr = ms.get("completionRate", 0)
    total_ms = ms.get("total", 0)
    approved = ms.get("approved", 0)
    overdue  = ms.get("overdue", 0)
    with_logs = ms.get("withLogs", 0)

    # If evaluation already finalized, just return those scores
    if req.isFinalized and req.existingTotal is not None:
        es = req.existingScores or {}
        return {
            "prediction": {
                "isFinalized": True,
                "actualScore": req.existingTotal,
                "message": "This project has already been evaluated and finalized.",
                "scores": es,
                "confidence": "actual",
            }
        }

    # Predict each category (out of 25 each)
    # 1. Proposal Quality — based on project status + files
    proposal = 12  # baseline
    if req.projectStatus == "approved":   proposal += 8
    elif req.projectStatus == "completed": proposal += 10
    elif req.projectStatus == "pending":   proposal -= 3
    if req.filesCount >= 5:  proposal += 5
    elif req.filesCount >= 2: proposal += 2
    proposal = max(0, min(25, proposal))

    # 2. Progress & Effort — based on milestones + logs
    progress = 10  # baseline
    if cr >= 80:   progress += 12
    elif cr >= 60: progress += 8
    elif cr >= 40: progress += 4
    elif cr >= 20: progress += 1
    else:          progress -= 5
    if overdue == 0 and total_ms > 0: progress += 3
    elif overdue >= 3: progress -= 5
    log_ratio = (with_logs / total_ms) if total_ms > 0 else 0
    if log_ratio >= 0.8: progress += 3
    progress = max(0, min(25, progress))

    # 3. Report Quality — based on files + feedback
    report = 10  # baseline
    if req.filesCount >= 3:   report += 8
    elif req.filesCount >= 1: report += 4
    if req.feedbackCount >= 3: report += 5
    elif req.feedbackCount >= 1: report += 2
    report = max(0, min(25, report))

    # 4. Technical Skill — based on overall milestone completion
    technical = 10  # baseline
    if cr >= 80:   technical += 12
    elif cr >= 60: technical += 8
    elif cr >= 40: technical += 4
    if req.projectStatus == "completed": technical += 3
    technical = max(0, min(25, technical))

    total = proposal + progress + report + technical

    def grade(t):
        if t >= 90: return "A+"
        if t >= 80: return "A"
        if t >= 70: return "B+"
        if t >= 60: return "B"
        if t >= 50: return "C+"
        if t >= 40: return "C"
        if t >= 33: return "D"
        return "F"

    # Confidence based on how much data we have
    data_points = (1 if req.filesCount > 0 else 0) + \
                  (1 if req.feedbackCount > 0 else 0) + \
                  (1 if total_ms > 0 else 0) + \
                  (1 if cr > 0 else 0)
    confidence = "high" if data_points >= 3 else "medium" if data_points >= 2 else "low"

    predicted_scores = {
        "proposalQuality":    proposal,
        "progressAndEffort":  progress,
        "reportQuality":      report,
        "technicalSkill":     technical,
    }

    recommendations = []
    if proposal < 18:
        recommendations.append("Ensure project proposal is well-documented and status is approved.")
    if progress < 15:
        recommendations.append(f"Complete more weekly milestones — currently at {cr}% completion rate.")
    if report < 15:
        recommendations.append("Upload more project files and request feedback from supervisor.")
    if technical < 15:
        recommendations.append("Demonstrate stronger technical implementation through milestone log entries.")
    if overdue > 0:
        recommendations.append(f"Address {overdue} overdue milestone(s) immediately to improve score.")

    return {
        "prediction": {
            "isFinalized": False,
            "predictedScores": predicted_scores,
            "predictedTotal": total,
            "predictedGrade": grade(total),
            "confidence": confidence,
            "recommendations": recommendations,
            "message": f"Predicted score based on current project data. Confidence: {confidence}.",
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 10: AUTO-GENERATE EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

class EvalReportRequest(BaseModel):
    projectTitle: str
    projectStatus: str
    studentName: str
    supervisorName: str
    filesCount: int
    feedbackCount: int
    milestoneStats: dict
    evaluation: Optional[dict] = None
    milestones: Optional[list] = []

@app.post("/generate-eval-report")
def generate_eval_report(req: EvalReportRequest):
    ms = req.milestoneStats
    cr = ms.get("completionRate", 0)
    total_ms  = ms.get("total", 0)
    approved  = ms.get("approved", 0)
    overdue   = ms.get("overdue", 0)
    submitted = ms.get("submitted", 0)
    rejected  = ms.get("rejected", 0)
    hours     = ms.get("totalHoursLogged", 0)
    ev        = req.evaluation

    # Overall performance level
    if ev and ev.get("isFinalized"):
        score = ev.get("totalScore", 0)
        grade = ev.get("grade", "N/A")
        perf_level = "Excellent" if score >= 80 else "Good" if score >= 60 else "Satisfactory" if score >= 40 else "Needs Improvement"
    else:
        score = None
        grade = "Not yet evaluated"
        perf_level = "Excellent" if cr >= 80 else "Good" if cr >= 60 else "Satisfactory" if cr >= 40 else "Needs Improvement"

    # Build milestone detail lines
    milestone_lines = []
    for m in (req.milestones or []):
        status_icon = {"approved": "✅", "submitted": "⏳", "rejected": "❌", "active": "🔄", "upcoming": "📅"}.get(m.get("status",""), "•")
        milestone_lines.append(
            f"  Week {m.get('weekNumber','?')}: {m.get('title','Untitled')} — {status_icon} {m.get('status','').capitalize()} "
            f"({m.get('logCount',0)} log entries, {m.get('hoursLogged',0)}h logged)"
        )
    milestone_section = "\n".join(milestone_lines) if milestone_lines else "  No milestones recorded."

    # Score section
    if ev and ev.get("isFinalized"):
        sc = ev.get("scores", {})
        score_section = f"""EVALUATION SCORES (Finalized)
  Proposal Quality    : {sc.get('proposalQuality', 0)}/25
  Progress & Effort   : {sc.get('progressAndEffort', 0)}/25
  Report Quality      : {sc.get('reportQuality', 0)}/25
  Technical Skill     : {sc.get('technicalSkill', 0)}/25
  ─────────────────────────────
  Total Score         : {ev.get('totalScore', 0)}/100
  Grade               : {ev.get('grade', 'N/A')}
  Remarks             : {ev.get('remarks', 'None')}"""
    elif ev:
        sc = ev.get("scores", {})
        score_section = f"""EVALUATION SCORES (Draft — Not Finalized)
  Proposal Quality    : {sc.get('proposalQuality', 0)}/25
  Progress & Effort   : {sc.get('progressAndEffort', 0)}/25
  Report Quality      : {sc.get('reportQuality', 0)}/25
  Technical Skill     : {sc.get('technicalSkill', 0)}/25
  ─────────────────────────────
  Total Score         : {ev.get('totalScore', 0)}/100  (DRAFT)
  Grade               : {ev.get('grade', 'N/A')}"""
    else:
        score_section = "EVALUATION SCORES\n  Not yet evaluated by supervisor."

    # Strengths & areas for improvement
    strengths = []
    improvements = []

    if cr >= 70:
        strengths.append(f"Strong milestone completion rate ({cr}%)")
    else:
        improvements.append(f"Milestone completion rate is {cr}% — needs improvement")

    if req.filesCount >= 3:
        strengths.append(f"Good documentation — {req.filesCount} files uploaded")
    elif req.filesCount == 0:
        improvements.append("No project files uploaded")

    if overdue == 0 and total_ms > 0:
        strengths.append("All milestones submitted on time (no overdue)")
    elif overdue > 0:
        improvements.append(f"{overdue} weekly milestone(s) are overdue")

    if req.feedbackCount >= 2:
        strengths.append(f"Active supervisor engagement ({req.feedbackCount} feedbacks)")
    else:
        improvements.append("Limited supervisor feedback — schedule more meetings")

    if hours >= 20:
        strengths.append(f"Significant effort logged ({hours} hours across milestones)")

    if rejected > 0:
        improvements.append(f"{rejected} milestone(s) were rejected — quality concerns noted")

    strengths_text    = "\n".join(f"  + {s}" for s in strengths) or "  None identified."
    improvements_text = "\n".join(f"  - {i}" for i in improvements) or "  None identified."

    import datetime as dt
    report_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║           ACADEMIC PROJECT EVALUATION REPORT                     ║
║           Generated by AI — FYP Management System               ║
╚══════════════════════════════════════════════════════════════════╝

REPORT DATE     : {dt.datetime.now().strftime('%d %B %Y, %H:%M')}
PROJECT TITLE   : {req.projectTitle}
STUDENT         : {req.studentName}
SUPERVISOR      : {req.supervisorName}
PROJECT STATUS  : {req.projectStatus.upper()}
PERFORMANCE     : {perf_level}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MILESTONE SUMMARY
  Total Milestones    : {total_ms}
  Approved            : {approved}
  Submitted           : {submitted}
  Rejected            : {rejected}
  Overdue             : {overdue}
  Completion Rate     : {cr}%
  Total Hours Logged  : {hours} hours

WEEKLY MILESTONE DETAIL
{milestone_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{score_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRENGTHS
{strengths_text}

AREAS FOR IMPROVEMENT
{improvements_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPERVISOR RECOMMENDATION
  Based on the data available, this student has demonstrated a
  {perf_level.lower()} level of engagement with their Final Year Project.
  {"The evaluation has been finalized with a grade of " + grade + "." if ev and ev.get("isFinalized") else "A formal evaluation is recommended at the earliest."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  This report was auto-generated by the AI Evaluation Engine.
  For official use, supervisor must review and sign off manually.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

    return {"success": True, "report": report_text}
