"""Rule-based ATS scoring engine.

Total = 100 marks, split section-wise:
  Contact 10 | Summary 10 | Experience 25 | Education 10 | Skills 15
  Formatting 15 | Keywords 15
"""
import re
from collections import Counter
from typing import Optional

from .parser import ParsedResume

# ----------------------------------------------------------------- lexicons
HEADING_ALIASES = {
    "summary": {"summary", "professional summary", "profile", "objective", "career objective",
                "about me", "about", "career summary", "executive summary", "personal statement"},
    "experience": {"experience", "work experience", "professional experience", "employment history",
                   "work history", "employment", "internships", "internship experience",
                   "relevant experience", "career history"},
    "education": {"education", "academic background", "academics", "education and training",
                  "academic qualifications", "qualifications"},
    "skills": {"skills", "technical skills", "key skills", "core competencies", "skills summary",
               "competencies", "technologies", "tech stack", "areas of expertise"},
    "projects": {"projects", "personal projects", "academic projects", "key projects"},
    "certifications": {"certifications", "certificates", "licenses", "courses",
                       "licenses and certifications", "training"},
    "achievements": {"achievements", "awards", "honors", "accomplishments", "awards and honors"},
}
ALIAS_TO_KEY = {a: k for k, v in HEADING_ALIASES.items() for a in v}

ACTION_VERBS = set("""
achieved administered analyzed analysed architected automated built championed collaborated
completed configured consolidated coordinated created cut debugged decreased delivered deployed
designed developed directed drove eliminated enabled engineered enhanced established evaluated
executed expanded facilitated finalized generated guided identified implemented improved increased
initiated integrated introduced launched led maintained managed mentored migrated modernized
monitored negotiated optimized optimised orchestrated organized organised oversaw owned partnered
performed pioneered planned presented produced programmed proposed reduced refactored released
remodeled researched resolved restructured reviewed scaled secured shipped simplified solved
spearheaded standardized streamlined strengthened supervised supported tested trained transformed
troubleshot upgraded wrote won accelerated acquired advised assessed authored boosted calculated
captured clarified conducted contributed converted customized defined demonstrated detected
diagnosed documented earned forecasted formulated founded headed influenced inspected instituted
investigated judged mapped measured mobilized operated prioritized prototyped reengineered
recruited reported revamped revised saved scheduled selected sourced taught tracked translated
unified validated visualized
""".split())

WEAK_PHRASES = ["responsible for", "worked on", "helped with", "helped to", "duties included",
                "tasked with", "assisted in", "was involved in", "participated in", "in charge of"]

PRONOUNS = re.compile(r"\b(i|me|my|myself|mine)\b", re.I)

STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or that the to was were will with
you your we our they their this these those who which what when where how about above after all
also any can could do does each etc more most must new not other out over per should such than
then there through into up use used using via work working ability able strong excellent good
experience years year team teams role responsibilities requirements required preferred plus
including include includes looking join company candidate position job description skills
knowledge understanding across within both well must may one two three based related ensure
""".split())

_RAW_SKILLS = """python,java,javascript,typescript,c++,c#,go,golang,rust,ruby,php,swift,kotlin,scala,sql,
nosql,matlab,html,css,react,angular,vue,node.js,express,django,flask,fastapi,spring,.net,next.js,
tailwind,bootstrap,redux,graphql,rest api,microservices,aws,azure,gcp,docker,kubernetes,terraform,
ansible,jenkins,ci/cd,git,github,gitlab,linux,bash,agile,scrum,jira,devops,mysql,postgresql,mongodb,
redis,elasticsearch,kafka,rabbitmq,spark,hadoop,airflow,pandas,numpy,scikit-learn,tensorflow,pytorch,
keras,nlp,machine learning,deep learning,data analysis,data science,tableau,power bi,excel,statistics,
etl,data warehousing,snowflake,bigquery,selenium,pytest,junit,unit testing,tdd,oop,design patterns,
system design,cloud,security,oauth,jwt,figma,ui/ux,seo,sap,salesforce,project management,stakeholder
management,communication,leadership,problem solving,collaboration,budgeting,forecasting,accounting,
auditing,marketing,analytics,crm,content strategy,sales,customer service,research,presentation"""
KNOWN_SKILLS = sorted({s.strip() for s in _RAW_SKILLS.replace("\n", "").split(",") if s.strip()})

BULLET_RE = re.compile(r"^\s*[\u2022\u25cf\u25aa\u25e6\u2023\u25ba\u27a2\u2713\u00b7\-\*\u2013▪►➢]+\s*")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}")
MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_RANGE_RE = re.compile(
    rf"(?:{MONTH}\s*)?\d{{4}}\s*(?:-|–|—|to)\s*(?:(?:present|current|now|ongoing)|(?:{MONTH}\s*)?\d{{4}})",
    re.I,
)
METRIC_RE = re.compile(r"\d+\s?%|[$₹€£]\s?\d|\b\d+(?:\.\d+)?\s?[kKmMbBxX]\b|\b\d{2,}\b")
DEGREE_RE = re.compile(
    r"\b(b\.?tech|m\.?tech|b\.?e\b|m\.?e\b|b\.?sc|m\.?sc|b\.?a\b|m\.?a\b|bca|mca|mba|bba|phd|ph\.d|"
    r"bachelor|master|diploma|associate|b\.?com|m\.?com|high school|secondary|12th|10th)",
    re.I,
)
INSTITUTION_RE = re.compile(r"\b(university|college|institute|school|academy|iit|nit|polytechnic)\b", re.I)


# ------------------------------------------------------------------ helpers
def _norm_heading(line: str) -> str:
    return re.sub(r"[^a-z& ]", "", line.lower().replace("&", "and")).strip()


def split_sections(text: str):
    sections, current, found = {}, "header", []
    sections[current] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = ALIAS_TO_KEY.get(_norm_heading(line)) if len(line) <= 40 else None
        if key:
            current = key
            found.append(key)
            sections.setdefault(current, [])
            continue
        sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}, found


def _has_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![\w+#.]){re.escape(term)}(?![\w+#])", text, re.I) is not None


def _ratio_points(ratio: float, target: float, max_pts: float) -> float:
    return round(max_pts * min(ratio / target, 1.0), 1)


class _Section:
    def __init__(self, name: str, max_score: int):
        self.name, self.max, self.score = name, max_score, 0.0
        self.issues, self.passed = [], []

    def add(self, pts: float):
        self.score += pts

    def ok(self, msg: str):
        self.passed.append(msg)

    def issue(self, severity: str, msg: str, fix: str):
        self.issues.append({"severity": severity, "message": msg, "fix": fix})

    def to_dict(self):
        score = round(min(self.score, self.max), 1)
        return {
            "name": self.name,
            "score": score,
            "max": self.max,
            "percent": round(score / self.max * 100),
            "issues": self.issues,
            "passed": self.passed,
        }


# ------------------------------------------------------------ section checks
def _check_contact(header_text: str, full_text: str) -> _Section:
    s = _Section("Contact information", 10)
    top = "\n".join(full_text.splitlines()[:15])
    if EMAIL_RE.search(top):
        s.add(3); s.ok("Email address found.")
    else:
        s.issue("high", "No email address found near the top.", "Add a professional email (name@domain.com) in the header.")
    phone = next((m for m in PHONE_RE.findall(top) if len(re.sub(r"\D", "", m)) >= 10), None)
    if phone:
        s.add(3); s.ok("Phone number found.")
    else:
        s.issue("high", "No phone number found.", "Add a phone number with country code, e.g. +91 98765 43210.")
    if re.search(r"linkedin\.com", full_text, re.I):
        s.add(2); s.ok("LinkedIn profile linked.")
    else:
        s.issue("medium", "LinkedIn profile is missing.", "Add your LinkedIn URL (linkedin.com/in/your-name).")
    if re.search(r"github\.com|gitlab\.com|behance\.net|portfolio|https?://", full_text, re.I):
        s.add(1); s.ok("Portfolio / GitHub link found.")
    else:
        s.issue("low", "No portfolio or GitHub link.", "Link a portfolio, GitHub or relevant work samples.")
    first = next((l.strip() for l in full_text.splitlines() if l.strip()), "")
    if 1 < len(first.split()) <= 5 and re.fullmatch(r"[A-Za-z.'\- ]+", first):
        s.add(1); s.ok("Name appears at the top.")
    else:
        s.issue("low", "Your name isn't clearly the first line.", "Put your full name alone on the first line.")
    return s


def _check_summary(sections: dict) -> _Section:
    s = _Section("Professional summary", 10)
    text = sections.get("summary", "")
    if not text:
        s.issue("medium", "No summary or profile section.", "Add a 2-3 line summary with your role, years of experience and top skills.")
        return s
    s.add(4); s.ok("Summary section present.")
    words = len(text.split())
    if 30 <= words <= 100:
        s.add(4); s.ok(f"Good summary length ({words} words).")
    else:
        s.add(2)
        s.issue("low", f"Summary is {words} words.", "Aim for 30-100 words: concise but specific.")
    if PRONOUNS.search(text):
        s.issue("low", "Summary uses first-person pronouns (I, my).", "Write in implied first person: 'Backend engineer with 4 years...'.")
    else:
        s.add(2); s.ok("Summary avoids personal pronouns.")
    return s


def _experience_bullets(exp: str):
    lines = [l for l in exp.splitlines() if l.strip()]
    marked = [BULLET_RE.sub("", l).strip() for l in lines if BULLET_RE.match(l)]
    if marked:
        return marked, True
    return [l for l in lines if len(l.split()) >= 6], False


def _check_experience(sections: dict) -> _Section:
    s = _Section("Work experience", 25)
    exp = sections.get("experience", "")
    if not exp:
        s.issue("high", "No work experience section detected.",
                "Add a heading named 'Work Experience' or 'Professional Experience'. Freshers can use 'Internships' or 'Projects'.")
        return s
    s.add(5); s.ok("Experience section present.")

    dates = DATE_RANGE_RE.findall(exp)
    if len(dates) >= 1:
        s.add(5); s.ok(f"Employment dates found ({len(dates)} date ranges).")
    else:
        s.issue("high", "No clear employment date ranges.", "Use a consistent format such as 'Jan 2022 - Present' beside every role.")

    bullets, marked = _experience_bullets(exp)
    if len(bullets) >= 4:
        s.add(4); s.ok(f"{len(bullets)} achievement lines detected.")
    else:
        s.issue("high", "Too few bullet points under experience.", "Use 3-5 bullets per role describing what you did and achieved.")
    if not marked and bullets:
        s.issue("low", "Bullets aren't marked with standard symbols.", "Use simple round bullets (•) so ATS parsers split achievements properly.")

    if bullets:
        starts = [re.sub(r"[^a-z]", "", b.split()[0].lower()) for b in bullets if b.split()]
        verb_ratio = sum(1 for w in starts if w in ACTION_VERBS) / max(len(starts), 1)
        s.add(_ratio_points(verb_ratio, 0.6, 5))
        if verb_ratio >= 0.6:
            s.ok(f"{round(verb_ratio * 100)}% of bullets start with strong action verbs.")
        else:
            weak = [b for b, w in zip(bullets, starts) if w not in ACTION_VERBS][:3]
            s.issue("medium", f"Only {round(verb_ratio * 100)}% of bullets start with action verbs.",
                    "Start bullets with verbs like Built, Led, Reduced, Automated. Examples to rewrite: "
                    + " | ".join(f"'{w[:60]}'" for w in weak))

        metric_ratio = sum(1 for b in bullets if METRIC_RE.search(b)) / len(bullets)
        s.add(_ratio_points(metric_ratio, 0.4, 6))
        if metric_ratio >= 0.4:
            s.ok(f"{round(metric_ratio * 100)}% of bullets include measurable results.")
        else:
            s.issue("high", f"Only {round(metric_ratio * 100)}% of bullets are quantified.",
                    "Add numbers: % improvement, time saved, users served, revenue, team size.")

        lowered = exp.lower()
        found_weak = [p for p in WEAK_PHRASES if p in lowered]
        if found_weak:
            s.issue("medium", "Weak phrases found: " + ", ".join(f"'{p}'" for p in found_weak),
                    "Replace passive phrases with direct ownership: 'Built...', 'Led...', 'Delivered...'.")
        long_ones = [b for b in bullets if len(b.split()) > 35]
        if long_ones:
            s.issue("low", f"{len(long_ones)} bullet(s) are longer than 35 words.", "Keep bullets to 1-2 lines (15-30 words).")
    return s


def _check_education(sections: dict) -> _Section:
    s = _Section("Education", 10)
    edu = sections.get("education", "")
    if not edu:
        s.issue("high", "No education section detected.", "Add an 'Education' heading with degree, institution and year.")
        return s
    s.add(4); s.ok("Education section present.")
    for pattern, pts, ok_msg, sev, bad, fix in [
        (DEGREE_RE, 3, "Degree/qualification mentioned.", "medium", "No recognisable degree found.", "Write the full degree, e.g. 'B.Tech in Computer Science'."),
        (re.compile(r"\b(19|20)\d{2}\b"), 2, "Graduation year found.", "medium", "No graduation year.", "Add the graduation (or expected) year."),
        (INSTITUTION_RE, 1, "Institution named.", "low", "Institution name not clear.", "Include the university / college name."),
    ]:
        if pattern.search(edu):
            s.add(pts); s.ok(ok_msg)
        else:
            s.issue(sev, bad, fix)
    return s


def _skill_items(skills_text: str):
    items = []
    for line in skills_text.splitlines():
        line = BULLET_RE.sub("", line)
        if ":" in line:
            line = line.split(":", 1)[1]
        for part in re.split(r"[,|;/•]", line):
            part = part.strip()
            if 0 < len(part.split()) <= 4:
                items.append(part)
    return items


def _check_skills(sections: dict, text: str) -> _Section:
    s = _Section("Skills", 15)
    sk = sections.get("skills", "")
    if sk:
        s.add(5); s.ok("Dedicated skills section present.")
        n = len(_skill_items(sk))
        s.add(_ratio_points(n, 8, 5))
        if n >= 8:
            s.ok(f"{n} skills listed.")
        else:
            s.issue("medium", f"Only {n} skills listed.", "List at least 8-12 relevant hard skills and tools.")
    else:
        s.issue("high", "No skills section detected.", "Add a 'Skills' section; ATS filters search it first.")
    hits = [k for k in KNOWN_SKILLS if _has_term(text, k)]
    s.add(_ratio_points(len(hits), 10, 5))
    if len(hits) >= 10:
        s.ok(f"{len(hits)} recognised industry skills across the resume.")
    else:
        s.issue("medium", f"Only {len(hits)} recognised industry skills found.", "Mention tools, technologies and methods by their standard names.")
    return s


def _check_formatting(parsed: ParsedResume, sections: dict, found: list, words: int) -> _Section:
    s = _Section("Formatting & length", 15)
    if 350 <= words <= 900:
        s.add(4); s.ok(f"Length is healthy ({words} words).")
    elif words < 350:
        s.add(1); s.issue("high", f"Resume is thin ({words} words).", "Expand with achievements, projects and skills (aim for 400-800 words).")
    else:
        s.add(2); s.issue("medium", f"Resume is long ({words} words).", "Trim to the most relevant 2 pages max.")
    if parsed.pages <= 2:
        s.add(3); s.ok(f"{parsed.pages} page(s): ideal length.")
    else:
        s.issue("medium", f"Resume runs {parsed.pages} pages.", "Keep it to 1 page (under 8 yrs experience) or 2 pages.")
    core = {"experience", "education", "skills"} & set(found)
    std = len(set(found))
    if std >= 4:
        s.add(4); s.ok("Standard section headings detected.")
    else:
        s.add(round(std))
        s.issue("medium", "Few standard section headings recognised.", "Use classic headings: Summary, Experience, Education, Skills, Projects.")
    if parsed.tables or parsed.images:
        s.issue("medium", f"Detected {parsed.tables} table(s) and {parsed.images} image(s).",
                "Tables, columns and images often scramble ATS parsing. Use a single-column text layout.")
    else:
        s.add(2); s.ok("No tables or images that confuse ATS parsers.")
    odd = sum(1 for c in parsed.text if ord(c) > 0x2FFF or c in "\ufffd□■")
    if odd / max(len(parsed.text), 1) > 0.01:
        s.issue("medium", "Unusual symbols or icons detected.", "Replace icon fonts and special glyphs with plain text.")
    else:
        s.add(1); s.ok("Characters are plain and parseable.")
    bullets = len(re.findall(r"^\s*[•●▪◦\-\*]", parsed.text, re.M))
    if bullets >= 5:
        s.add(1); s.ok("Bullet points used for scannability.")
    else:
        s.issue("low", "Very few bullet points overall.", "Break paragraphs into short bullet points.")
    return s


def _jd_keywords(jd: str, top: int = 25):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-/]{1,}", jd.lower())
    tokens = [t.strip(".-/") for t in tokens]
    counts = Counter(t for t in tokens if t not in STOPWORDS and len(t) > 2)
    keys = [k for k, _ in counts.most_common(top * 2)]
    for sk in KNOWN_SKILLS:  # always include known skills the JD mentions
        if _has_term(jd, sk) and sk not in keys:
            keys.append(sk)
    return keys[: top + 10]


def _check_keywords(text: str, jd: Optional[str]) -> _Section:
    s = _Section("Keyword match", 15)
    if jd and jd.strip():
        keys = _jd_keywords(jd)
        matched = [k for k in keys if _has_term(text, k)]
        missing = [k for k in keys if k not in matched]
        ratio = len(matched) / max(len(keys), 1)
        s.add(_ratio_points(ratio, 0.7, 15))
        if ratio >= 0.7:
            s.ok(f"Matched {len(matched)} of {len(keys)} job-description keywords.")
        else:
            s.issue("high", f"Matched only {len(matched)} of {len(keys)} job-description keywords ({round(ratio * 100)}%).",
                    "Work these naturally into skills and bullets: " + ", ".join(missing[:12]))
        s.matched, s.missing = matched, missing
    else:
        hits = [k for k in KNOWN_SKILLS if _has_term(text, k)]
        s.add(_ratio_points(len(hits), 12, 12))
        s.ok(f"Found {len(hits)} common industry keywords.")
        s.issue("low", "No job description supplied, so keyword score is generic (max 12/15).",
                "Paste the target job description to get an exact keyword match and missing-keyword list.")
        s.matched, s.missing = hits, []
    return s


# ---------------------------------------------------------------- public API
def analyze(parsed: ParsedResume, job_description: Optional[str] = None) -> dict:
    text = parsed.text
    words = len(text.split())

    if words < 40:
        return {
            "overall_score": 0, "grade": "Unreadable", "sections": [], "stats": {"words": words, "pages": parsed.pages},
            "summary": "Almost no text could be extracted. This is usually a scanned image or a PDF made from outlines, "
                       "which ATS software cannot read. Export a text-based PDF or DOCX and try again.",
            "top_fixes": [{"severity": "high", "message": "Resume text is not machine-readable.",
                           "fix": "Re-export from Word/Google Docs as PDF (don't print-to-image or scan)."}],
            "keywords": {"matched": [], "missing": []},
        }

    sections, found = split_sections(text)
    checks = [
        _check_contact(sections.get("header", ""), text),
        _check_summary(sections),
        _check_experience(sections),
        _check_education(sections),
        _check_skills(sections, text),
        _check_formatting(parsed, sections, found, words),
        _check_keywords(text, job_description),
    ]
    kw = checks[-1]
    result = [c.to_dict() for c in checks]
    total = round(sum(c["score"] for c in result))

    grade = ("Excellent" if total >= 85 else "Good" if total >= 70
             else "Needs work" if total >= 50 else "Poor")
    order = {"high": 0, "medium": 1, "low": 2}
    all_issues = [dict(i, section=c["name"]) for c in result for i in c["issues"]]
    top = sorted(all_issues, key=lambda i: order[i["severity"]])[:5]
    weakest = min(result, key=lambda c: c["percent"])
    return {
        "overall_score": total,
        "grade": grade,
        "summary": f"Your resume scores {total}/100 ({grade.lower()}). "
                   f"The weakest area is {weakest['name'].lower()} at {weakest['score']}/{weakest['max']}.",
        "sections": result,
        "top_fixes": top,
        "keywords": {"matched": getattr(kw, "matched", [])[:40], "missing": getattr(kw, "missing", [])[:30]},
        "stats": {
            "words": words,
            "pages": parsed.pages,
            "sections_detected": sorted(set(found)),
            "issue_count": len(all_issues),
        },
    }
