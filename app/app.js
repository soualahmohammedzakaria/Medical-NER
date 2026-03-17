/* ==========================================================
   Medical NER – Application Logic
   ========================================================== */

const API_URL = "https://erp-clinic.58wilaya.com/ai";

// ---- DOM refs ----
const textInput     = document.getElementById("textInput");
const analyzeBtn    = document.getElementById("analyzeBtn");
const clearBtn      = document.getElementById("clearBtn");
const fileInput     = document.getElementById("fileInput");
const uploadArea    = document.getElementById("uploadArea");
const fileNameHint  = document.getElementById("fileName");
const resultsCard   = document.getElementById("resultsCard");
const annotatedText = document.getElementById("annotatedText");
const entitySummary = document.getElementById("entitySummary");
const errorBanner   = document.getElementById("errorBanner");
const errorMessage  = document.getElementById("errorMessage");
const errorClose    = document.getElementById("errorClose");
const btnText       = analyzeBtn.querySelector(".btn__text");
const btnLoader     = analyzeBtn.querySelector(".btn__loader");

// ---- File Upload ----
uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();

  if (ext === "txt") {
    const text = await file.text();
    textInput.value = text;
    fileNameHint.textContent = `Loaded: ${file.name}`;
  } else if (ext === "pdf") {
    try {
      const text = await extractPdfText(file);
      textInput.value = text;
      fileNameHint.textContent = `Loaded: ${file.name}`;
    } catch (err) {
      showError("Failed to read PDF file. Please try a different file or paste the text manually.");
      console.error(err);
    }
  } else {
    showError("Unsupported file type. Please upload a .txt or .pdf file.");
  }
}

async function extractPdfText(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdfjsLib = window.pdfjsLib;
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

  const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(arrayBuffer) });
  const pdf = await loadingTask.promise;
  const pages = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map((item) => item.str);
    pages.push(strings.join(" "));
  }

  return pages.join("\n\n");
}

// ---- Analyze ----
analyzeBtn.addEventListener("click", analyze);

async function analyze() {
  const text = textInput.value.trim();
  if (!text) {
    showError("Please enter or upload some text to analyse.");
    return;
  }

  setLoading(true);
  hideError();
  resultsCard.hidden = true;

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.message || `Server responded with ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
    console.error(err);
  } finally {
    setLoading(false);
  }
}

// ---- Render Results ----
function renderResults(data) {
  const { text, entities } = data;

  // Sort entities by start offset
  const sorted = [...entities].sort((a, b) => a.start - b.start);

  // Build annotated HTML
  let html = "";
  let cursor = 0;

  for (const ent of sorted) {
    // Skip overlapping entities
    if (ent.start < cursor) continue;

    // Text before entity
    if (ent.start > cursor) {
      html += escapeHtml(text.slice(cursor, ent.start));
    }

    const cssClass = entityCssClass(ent.label);
    html += `<span class="entity-tag ${cssClass}">`;
    html += escapeHtml(ent.text);
    html += `<span class="entity-tag__label">${escapeHtml(ent.label)}</span>`;
    html += `</span>`;

    cursor = ent.end;
  }

  // Remaining text
  if (cursor < text.length) {
    html += escapeHtml(text.slice(cursor));
  }

  annotatedText.innerHTML = html;

  // Entity summary badges
  const counts = {};
  for (const ent of entities) {
    counts[ent.label] = (counts[ent.label] || 0) + 1;
  }

  entitySummary.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => {
      const cssClass = entityCssClass(label);
      return `<span class="entity-badge ${cssClass}">
        ${escapeHtml(label)}
        <span class="entity-badge__count">${count}</span>
      </span>`;
    })
    .join("");

  resultsCard.hidden = false;
  resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- Helpers ----
function entityCssClass(label) {
  const l = label.toLowerCase().replace(/[\s\/]/g, "_");
  const map = {
    disease: "entity-tag--disease",
    chemical: "entity-tag--chemical",
    drug: "entity-tag--chemical",
    gene: "entity-tag--gene",
    protein: "entity-tag--protein",
    species: "entity-tag--species",
    organism: "entity-tag--species",
    mutation: "entity-tag--mutation",
    cell_line: "entity-tag--cell_line",
    cell_type: "entity-tag--cell_type",
    cell: "entity-tag--cell",
    dna: "entity-tag--dna",
    rna: "entity-tag--rna",
  };
  return map[l] || "entity-tag--other";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnText.textContent = loading ? "Analysing…" : "Analyze";
  btnLoader.hidden = !loading;
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
}

// ---- Clear ----
clearBtn.addEventListener("click", () => {
  textInput.value = "";
  fileInput.value = "";
  fileNameHint.textContent = "";
  resultsCard.hidden = true;
  hideError();
});

errorClose.addEventListener("click", hideError);

// ---- Example Clinical Case ----
const EXAMPLE_TEXT = `CLINICAL CASE REPORT — Patient ID: 847291
Date: ${new Date().toLocaleDateString("en-GB")}

CHIEF COMPLAINT
The patient is a 58-year-old male presenting with progressive fatigue, polyuria,
and polydipsia over the past 3 months.

MEDICAL HISTORY
The patient has a known history of type 2 diabetes mellitus diagnosed in 2018,
managed with metformin 1000 mg twice daily. He also carries a diagnosis of
hypertension and hyperlipidemia, for which he takes lisinopril 10 mg and
atorvastatin 40 mg respectively.

CURRENT MEDICATIONS
- Metformin 1000 mg (oral, twice daily)
- Lisinopril 10 mg (oral, once daily)
- Atorvastatin 40 mg (oral, once daily at bedtime)
- Aspirin 81 mg (oral, once daily)

EXAMINATION FINDINGS
Blood pressure: 148/92 mmHg. Heart rate: 82 bpm. BMI: 31.4 kg/m2.
Peripheral neuropathy noted in bilateral lower extremities.
No signs of diabetic retinopathy on fundoscopic exam.

LABORATORY RESULTS
- HbA1c: 9.2% (target <7%)
- Fasting glucose: 218 mg/dL
- LDL cholesterol: 112 mg/dL
- eGFR: 64 mL/min/1.73m2 (mild chronic kidney disease)
- Urine albumin-to-creatinine ratio: 42 mg/g (microalbuminuria)

ASSESSMENT
Poorly controlled type 2 diabetes mellitus with peripheral neuropathy and
early diabetic nephropathy. Secondary hypertension likely contributing to
renal deterioration. Dyslipidemia partially controlled on current statin therapy.

PLAN
1. Add empagliflozin 10 mg once daily for glycemic control and renal protection.
2. Increase lisinopril to 20 mg for blood pressure and nephroprotection.
3. Refer to endocrinology for insulin initiation if HbA1c remains above 9%.
4. Ophthalmology referral for annual diabetic retinopathy screening.
5. Nutritional counseling for low-carbohydrate diet adherence.
6. Follow-up in 8 weeks with repeat HbA1c, BMP, and urine albumin.

SECONDARY DIAGNOSIS NOTE
The patient was also recently treated for a urinary tract infection caused by
Escherichia coli, for which he completed a 7-day course of ciprofloxacin 500 mg.
He reports full resolution of dysuria. No recurrent symptoms.

ALLERGIES
Penicillin (rash), Sulfonamides (unknown reaction)

Attending Physician: Dr. Sarah Mendez, MD
Department of Internal Medicine`;

document.getElementById("exampleBtn").addEventListener("click", () => {
  textInput.value = EXAMPLE_TEXT;
  fileNameHint.textContent = "";
  fileInput.value = "";
  resultsCard.hidden = true;
  hideError();
});
