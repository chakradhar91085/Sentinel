/* =============================================
   SENTINEL — Frontend Logic
   ============================================= */

// ── DOM References ──────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    mainCard:          $('#mainCard'),
    commentInput:      $('#commentInput'),
    charCount:         $('#charCount'),
    scanBtn:           $('#scanBtn'),
    btnContent:        $('#btnContent'),
    btnLoading:        $('#btnLoading'),
    resultsDivider:    $('#resultsDivider'),
    resultsSection:    $('#resultsSection'),
    barToxic:          $('#bar-toxic'),
    valToxic:          $('#val-toxic'),
    barHateSpeech:     $('#bar-hate_speech'),
    valHateSpeech:     $('#val-hate_speech'),
    barInsult:         $('#bar-insult'),
    valInsult:         $('#val-insult'),
    barThreat:         $('#bar-threat'),
    valThreat:         $('#val-threat'),
    barAbusive:        $('#bar-abusive'),
    valAbusive:        $('#val-abusive'),
    verdictCard:       $('#verdictCard'),
    verdictTitle:      $('#verdictTitle'),
    verdictDesc:       $('#verdictDesc'),
    confidenceValue:   $('#confidenceValue'),
    confidenceFill:    $('#confidenceFill'),
    historySection:    $('#historySection'),
    historyList:       $('#historyList'),
    clearHistoryBtn:   $('#clearHistoryBtn'),
    toast:             $('#toast'),
    toastMessage:      $('#toastMessage'),
    totalScans:        $('#totalScans'),
    avgScore:          $('#avgScore'),
    clearTextBtn:      $('#clearTextBtn'),
    copyResultBtn:     $('#copyResultBtn'),
    // Tabs
    tabScannerBtn:     $('#tab-scanner-btn'),
    tabAnalyticsBtn:   $('#tab-analytics-btn'),
    tabScanner:        $('#tab-scanner'),
    tabAnalytics:      $('#tab-analytics'),
    analyticsEmpty:    $('#analyticsEmpty'),
    dashboard:         $('#dashboard'),
    // Stat cards
    statTotal:         $('#stat-total'),
    statToxicCount:    $('#stat-toxic-count'),
    statAvg:           $('#stat-avg'),
    statPeak:          $('#stat-peak'),
};

// ── Config ──────────────────────────────────
const API_URL = window.location.protocol === 'file:'
    ? 'http://127.0.0.1:8000/predict'
    : '/predict';

const HISTORY_KEY = 'sentinel_history';
const MAX_HISTORY = 50;

// Chart.js shared defaults
const CHART_DEFAULTS = {
    color: 'rgba(232, 228, 223, 0.7)',
    borderColor: 'rgba(255,255,255,0.06)',
    font: { family: "'DM Sans', sans-serif", size: 11 },
};

// ── State ───────────────────────────────────
let isScanning = false;
let history = [];
let charts = {};

try {
    history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    // Drop any stale entries that are missing categories (would cause Object.values crash)
    history = history.filter(h => h && h.categories && typeof h.categories === 'object');
} catch { history = []; }


/* ===========================================
   TAB SWITCHING
   =========================================== */
function switchTab(tab) {
    if (tab === 'scanner') {
        DOM.tabScanner.classList.remove('hidden');
        DOM.tabAnalytics.classList.add('hidden');
        DOM.tabScannerBtn.classList.add('active');
        DOM.tabScannerBtn.setAttribute('aria-selected', 'true');
        DOM.tabAnalyticsBtn.classList.remove('active');
        DOM.tabAnalyticsBtn.setAttribute('aria-selected', 'false');
    } else {
        DOM.tabScanner.classList.add('hidden');
        DOM.tabAnalytics.classList.remove('hidden');
        DOM.tabScannerBtn.classList.remove('active');
        DOM.tabScannerBtn.setAttribute('aria-selected', 'false');
        DOM.tabAnalyticsBtn.classList.add('active');
        DOM.tabAnalyticsBtn.setAttribute('aria-selected', 'true');
        updateDashboard();
    }
}

DOM.tabScannerBtn.addEventListener('click', () => switchTab('scanner'));
DOM.tabAnalyticsBtn.addEventListener('click', () => switchTab('analytics'));


/* ===========================================
   MULTI-LABEL BAR HELPERS
   =========================================== */
function animateBar(barEl, valEl, prob) {
    const pct = Math.round(prob * 100);
    barEl.style.width = `${pct}%`;
    valEl.textContent = `${pct}%`;

    barEl.classList.remove('fill-safe', 'fill-warn', 'fill-danger');
    if (pct < 30) barEl.classList.add('fill-safe');
    else if (pct < 60) barEl.classList.add('fill-warn');
    else barEl.classList.add('fill-danger');
}


/* ===========================================
   VERDICT HELPERS
   =========================================== */
function getVerdict(prob) {
    const pct = prob * 100;
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
    ['barToxic', 'barHateSpeech', 'barInsult', 'barThreat', 'barAbusive'].forEach(key => {
        if (DOM[key]) DOM[key].style.width = '0%';
    });
    ['valToxic', 'valHateSpeech', 'valInsult', 'valThreat', 'valAbusive'].forEach(key => {
        if (DOM[key]) DOM[key].textContent = '0%';
    });
    DOM.confidenceFill.style.width = '0%';
}

function showResults(data, text) {
    if (!data || !data.categories) {
        showToast('Unexpected response from server.');
        return;
    }
    const maxProb = Math.max(...Object.values(data.categories));
    const confidence = Math.round(maxProb * 100);
    const verdict = getVerdict(maxProb);

    DOM.resultsDivider.classList.add('visible');
    DOM.resultsSection.classList.add('visible');

    DOM.verdictCard.className = `verdict-card verdict-${verdict.cls}`;
    DOM.verdictTitle.textContent = verdict.title;
    DOM.verdictDesc.textContent = verdict.desc;

    setTimeout(() => {
        animateBar(DOM.barToxic, DOM.valToxic, data.categories.toxic);
        animateBar(DOM.barHateSpeech, DOM.valHateSpeech, data.categories.hate_speech);
        animateBar(DOM.barInsult, DOM.valInsult, data.categories.insult);
        animateBar(DOM.barThreat, DOM.valThreat, data.categories.threat);
        animateBar(DOM.barAbusive, DOM.valAbusive, data.categories.abusive);

        DOM.confidenceValue.textContent = `${confidence}%`;
        DOM.confidenceFill.style.width = `${confidence}%`;
        DOM.confidenceFill.className = `confidence-fill confidence-${verdict.cls}`;
    }, 250);
}


/* ===========================================
   HISTORY
   =========================================== */
function addToHistory(text, data) {
    if (!data || !data.categories) return;
    const maxProb = Math.max(...Object.values(data.categories));
    history.unshift({
        text: text.substring(0, 120),
        fullText: text,
        toxicity: maxProb,
        label: data.is_toxic ? 'toxic' : 'non-toxic',
        categories: data.categories,
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
   STATS (Header)
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
   ANALYTICS DASHBOARD
   =========================================== */
function getChartColors() {
    return {
        line:    { border: '#c9785d', bg: 'rgba(201,120,93,0.12)' },
        safe:    { border: '#5a9a6e', bg: 'rgba(90,154,110,0.18)' },
        danger:  { border: '#c75454', bg: 'rgba(199,84,84,0.18)' },
        toxic:   '#c75454',
        hate:    '#a855f7',
        insult:  '#f97316',
        threat:  '#ef4444',
        abusive: '#c4943d',
    };
}

function buildChartDefaults() {
    Chart.defaults.color = 'rgba(154,147,137,0.9)';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
    Chart.defaults.font.family = "'DM Sans', sans-serif";
    Chart.defaults.font.size = 11;
}

function destroyChart(key) {
    if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

function updateDashboard() {
    buildChartDefaults();
    const c = getChartColors();

    if (history.length === 0) {
        DOM.analyticsEmpty.classList.remove('hidden');
        DOM.dashboard.classList.add('hidden');
        return;
    }

    DOM.analyticsEmpty.classList.add('hidden');
    DOM.dashboard.classList.remove('hidden');

    const recent = history.slice(0, 20).reverse();
    const toxicCount = history.filter(h => h.toxicity >= 0.5).length;
    const safeCount = history.length - toxicCount;
    const avgTox = history.reduce((s, h) => s + h.toxicity, 0) / history.length;
    const peakTox = Math.max(...history.map(h => h.toxicity));

    // Summary stats
    DOM.statTotal.textContent = history.length;
    DOM.statToxicCount.textContent = toxicCount;
    DOM.statAvg.textContent = `${Math.round(avgTox * 100)}%`;
    DOM.statPeak.textContent = `${Math.round(peakTox * 100)}%`;

    // Category averages
    const cats = ['toxic', 'hate_speech', 'insult', 'threat', 'abusive'];
    const catAvgs = cats.map(cat => {
        const vals = history.filter(h => h.categories).map(h => h.categories[cat] || 0);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    });

    // ── Chart 1: Line — Toxicity Over Time ──
    destroyChart('line');
    charts.line = new Chart($('#lineChart'), {
        type: 'line',
        data: {
            labels: recent.map((_, i) => `#${i + 1}`),
            datasets: [{
                label: 'Toxicity',
                data: recent.map(h => Math.round(h.toxicity * 100)),
                borderColor: c.line.border,
                backgroundColor: c.line.bg,
                borderWidth: 2,
                pointBackgroundColor: recent.map(h => h.toxicity >= 0.5 ? c.toxic : c.safe.border),
                pointRadius: 4,
                pointHoverRadius: 6,
                tension: 0.4,
                fill: true,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } } },
            scales: {
                y: {
                    min: 0, max: 100,
                    ticks: { callback: v => `${v}%`, stepSize: 25 },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
                x: { grid: { display: false } },
            },
        },
    });

    // ── Chart 2: Doughnut — Distribution ──
    destroyChart('doughnut');
    charts.doughnut = new Chart($('#doughnutChart'), {
        type: 'doughnut',
        data: {
            labels: ['Safe', 'Toxic'],
            datasets: [{
                data: [safeCount, toxicCount],
                backgroundColor: [c.safe.bg, c.danger.bg],
                borderColor: [c.safe.border, c.danger.border],
                borderWidth: 2,
                hoverOffset: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyleWidth: 8 } },
                tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} scans` } },
            },
        },
    });

    // ── Chart 3: Radar — Category Breakdown ──
    destroyChart('radar');
    charts.radar = new Chart($('#radarChart'), {
        type: 'radar',
        data: {
            labels: ['Toxic', 'Hate Speech', 'Insult', 'Threat', 'Abusive'],
            datasets: [{
                label: 'Avg Score',
                data: catAvgs.map(v => Math.round(v * 100)),
                backgroundColor: 'rgba(201,120,93,0.15)',
                borderColor: '#c9785d',
                borderWidth: 2,
                pointBackgroundColor: '#c9785d',
                pointRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } } },
            scales: {
                r: {
                    min: 0, max: 100,
                    ticks: { stepSize: 25, callback: v => `${v}%`, backdropColor: 'transparent' },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { font: { size: 11 } },
                },
            },
        },
    });

    // ── Chart 4: Bar — Category Intensity ──
    const catLabels = ['Toxic', 'Hate Speech', 'Insult', 'Threat', 'Abusive'];
    const catColors = [c.toxic, c.hate, c.insult, c.threat, c.abusive];
    destroyChart('bar');
    charts.bar = new Chart($('#barChart'), {
        type: 'bar',
        data: {
            labels: catLabels,
            datasets: [{
                label: 'Avg %',
                data: catAvgs.map(v => Math.round(v * 100)),
                backgroundColor: catColors.map(col => col + '33'),
                borderColor: catColors,
                borderWidth: 2,
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } } },
            scales: {
                y: {
                    min: 0, max: 100,
                    ticks: { callback: v => `${v}%`, stepSize: 25 },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
                x: { grid: { display: false } },
            },
        },
    });
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
    renderHistory();
    updateStats();
    DOM.commentInput.focus();
});
