/* =============================================
   SENTINEL — Frontend Logic
   ============================================= */

// ── DOM References ──────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    mainCard:        $('#mainCard'),
    commentInput:    $('#commentInput'),
    charCount:       $('#charCount'),
    scanBtn:         $('#scanBtn'),
    btnContent:      $('#btnContent'),
    btnLoading:      $('#btnLoading'),
    resultsDivider:  $('#resultsDivider'),
    resultsSection:  $('#resultsSection'),
    gaugeFill:       $('#gaugeFill'),
    gaugePercent:    $('#gaugePercent'),
    verdictCard:     $('#verdictCard'),
    verdictTitle:    $('#verdictTitle'),
    verdictDesc:     $('#verdictDesc'),
    confidenceValue: $('#confidenceValue'),
    confidenceFill:  $('#confidenceFill'),
    historySection:  $('#historySection'),
    historyList:     $('#historyList'),
    clearHistoryBtn: $('#clearHistoryBtn'),
    toast:           $('#toast'),
    toastMessage:    $('#toastMessage'),
    totalScans:      $('#totalScans'),
    avgScore:        $('#avgScore'),
    clearTextBtn:    $('#clearTextBtn'),
    copyResultBtn:   $('#copyResultBtn'),
};

// ── Config ──────────────────────────────────
const API_URL = window.location.protocol === 'file:'
    ? 'http://127.0.0.1:8000/predict'
    : '/predict';

const GAUGE_ARC_LENGTH = 267;
const HISTORY_KEY = 'sentinel_history';
const MAX_HISTORY = 20;

// ── State ───────────────────────────────────
let isScanning = false;
let history = [];

try {
    history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
} catch { history = []; }


/* ===========================================
   GAUGE HELPERS
   =========================================== */
function getGaugeColor(pct) {
    if (pct < 30) return { color: 'var(--color-safe)', cls: 'safe' };
    if (pct < 60) return { color: 'var(--color-warn)', cls: 'warn' };
    return { color: 'var(--color-danger)', cls: 'danger' };
}

function animateGauge(targetPct) {
    const offset = GAUGE_ARC_LENGTH * (1 - targetPct / 100);
    const color = getGaugeColor(targetPct);

    DOM.gaugeFill.style.stroke = color.color;
    DOM.gaugeFill.style.strokeDashoffset = offset;

    // Animate counter
    const duration = 1400;
    const start = performance.now();

    function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        DOM.gaugePercent.textContent = Math.round(eased * targetPct);
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}


/* ===========================================
   VERDICT HELPERS
   =========================================== */
function getVerdict(toxicity) {
    const pct = toxicity * 100;
    if (pct < 20) return { title: 'Safe Content', desc: 'This content appears to be safe and non-harmful.', cls: 'safe' };
    if (pct < 40) return { title: 'Mostly Safe', desc: 'This content is mostly safe with minimal concerns.', cls: 'safe' };
    if (pct < 60) return { title: 'Moderate Risk', desc: 'Moderate toxicity indicators detected. Review recommended.', cls: 'warn' };
    if (pct < 80) return { title: 'Toxic Content', desc: 'This content is likely toxic. Moderation action may be needed.', cls: 'danger' };
    return { title: 'Highly Toxic', desc: 'Highly toxic and harmful content. Immediate action recommended.', cls: 'danger' };
}


/* ===========================================
   CORE ANALYSIS
   =========================================== */
async function analyzeContent() {
    const text = DOM.commentInput.value.trim();
    if (!text || isScanning) return;

    isScanning = true;
    setLoading(true);
    resetResults();

    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const errBody = await res.json().catch(() => null);
            throw new Error(errBody?.detail || `API returned ${res.status}`);
        }

        const data = await res.json();

        // Brief pause so loading feels intentional
        await sleep(450);

        showResults(data, text);
        addToHistory(text, data);
        updateStats();

    } catch (err) {
        showToast(err.message || 'Failed to connect to Sentinel backend.');
    } finally {
        isScanning = false;
        setLoading(false);
    }
}


/* ===========================================
   UI FUNCTIONS
   =========================================== */
function setLoading(on) {
    DOM.scanBtn.disabled = on;
    DOM.btnContent.classList.toggle('hidden', on);
    DOM.btnLoading.classList.toggle('hidden', !on);
}

function resetResults() {
    DOM.gaugeFill.style.strokeDashoffset = GAUGE_ARC_LENGTH;
    DOM.gaugePercent.textContent = '0';
    DOM.confidenceFill.style.width = '0%';
}

function showResults(data, text) {
    const pct = Math.round(data.toxicity * 100);
    const confidence = Math.round(Math.abs(data.toxicity - 0.5) * 200);
    const verdict = getVerdict(data.toxicity);

    // Show divider + results
    DOM.resultsDivider.classList.add('visible');
    DOM.resultsSection.classList.add('visible');

    // Verdict card
    DOM.verdictCard.className = `verdict-card verdict-${verdict.cls}`;
    DOM.verdictTitle.textContent = verdict.title;
    DOM.verdictDesc.textContent = verdict.desc;

    // Gauge + confidence (delayed for entrance animation)
    setTimeout(() => {
        animateGauge(pct);
        DOM.confidenceValue.textContent = `${confidence}%`;
        DOM.confidenceFill.style.width = `${confidence}%`;
        DOM.confidenceFill.className = `confidence-fill confidence-${verdict.cls}`;
    }, 250);
}


/* ===========================================
   HISTORY
   =========================================== */
function addToHistory(text, data) {
    history.unshift({
        text: text.substring(0, 120),
        fullText: text,
        toxicity: data.toxicity,
        label: data.label,
        timestamp: new Date().toISOString(),
    });

    if (history.length > MAX_HISTORY) history.pop();

    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch {}
    renderHistory();
}

function renderHistory() {
    if (history.length === 0) {
        DOM.historySection.classList.remove('visible');
        return;
    }

    DOM.historySection.classList.add('visible');
    DOM.historyList.innerHTML = history.map((entry, i) => {
        const pct = Math.round(entry.toxicity * 100);
        const timeAgo = getTimeAgo(entry.timestamp);
        const colorCls = pct < 30 ? 'safe' : pct < 60 ? 'warn' : 'danger';

        return `
            <div class="history-item" data-index="${i}" title="Click to load this text">
                <p class="history-item-text">${escapeHtml(entry.text)}${entry.text.length >= 120 ? '…' : ''}</p>
                <div class="history-item-meta">
                    <span class="history-badge badge-${colorCls}">${pct}%</span>
                    <span class="history-time">${timeAgo}</span>
                </div>
            </div>
        `;
    }).join('');

    // Click to load text
    DOM.historyList.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            const idx = parseInt(item.dataset.index);
            DOM.commentInput.value = history[idx].fullText || history[idx].text;
            updateCharCount();
            DOM.commentInput.focus();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
}

function clearHistory() {
    history = [];
    try { localStorage.removeItem(HISTORY_KEY); } catch {}
    renderHistory();
    updateStats();
}


/* ===========================================
   STATS
   =========================================== */
function updateStats() {
    DOM.totalScans.textContent = history.length;
    if (history.length > 0) {
        const avg = history.reduce((sum, h) => sum + h.toxicity, 0) / history.length;
        DOM.avgScore.textContent = `${Math.round(avg * 100)}%`;
    } else {
        DOM.avgScore.textContent = '—';
    }
}


/* ===========================================
   UTILITIES
   =========================================== */
function updateCharCount() {
    DOM.charCount.textContent = DOM.commentInput.value.length.toLocaleString();
    if (DOM.clearTextBtn) {
        DOM.clearTextBtn.classList.toggle('hidden', DOM.commentInput.value.length === 0);
    }
}

function showToast(message) {
    DOM.toastMessage.textContent = message;
    DOM.toast.classList.add('show');
    setTimeout(() => DOM.toast.classList.remove('show'), 4500);
}

function escapeHtml(str) {
    const el = document.createElement('div');
    el.textContent = str;
    return el.innerHTML;
}

function getTimeAgo(iso) {
    const secs = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
    if (secs < 5) return 'just now';
    if (secs < 60) return `${secs}s ago`;
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}


/* ===========================================
   EVENT LISTENERS
   =========================================== */
DOM.scanBtn.addEventListener('click', analyzeContent);

DOM.commentInput.addEventListener('input', updateCharCount);

DOM.commentInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        analyzeContent();
    }
});

document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        DOM.commentInput.value = chip.dataset.text;
        updateCharCount();
        DOM.commentInput.focus();
    });
});

DOM.clearHistoryBtn.addEventListener('click', clearHistory);

if (DOM.clearTextBtn) {
    DOM.clearTextBtn.addEventListener('click', () => {
        DOM.commentInput.value = '';
        updateCharCount();
        DOM.commentInput.focus();
    });
}

if (DOM.copyResultBtn) {
    DOM.copyResultBtn.addEventListener('click', () => {
        const text = DOM.commentInput.value.trim();
        const verdict = DOM.verdictTitle.textContent;
        const confidence = DOM.confidenceValue.textContent;
        const resultText = `Sentinel Analysis:\nContent: "${text}"\nVerdict: ${verdict} (Confidence: ${confidence})`;
        
        navigator.clipboard.writeText(resultText).then(() => {
            showToast('Result copied to clipboard!');
        }).catch(() => {
            showToast('Failed to copy result.');
        });
    });
}


/* ===========================================
   INITIALIZE
   =========================================== */
document.addEventListener('DOMContentLoaded', () => {
    // History + Stats
    renderHistory();
    updateStats();

    // Focus
    DOM.commentInput.focus();
});
