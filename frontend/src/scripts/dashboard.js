// dashboard.js — Dashboard interactivity: source filtering, heatmap, ApexCharts trends, player card, share
// Loads metrics.json via fetch() at runtime — no build-time data injection.

import { initThemeToggle } from './theme.js';
import {
    CLIENT_ID_STORAGE_KEY,
    USERNAME_STORAGE_KEY,
    createStableClientId,
    findEntryRank,
    getSubmissionPayload as buildSubmissionPayload,
    sortLeaderboardEntries,
} from './leaderboard.js';

let sourceViews = {};
let defaultSource = 'both';
let activeSourceKey = 'both';
let activeView = null;
let branding = {};

const sourceFilter = document.getElementById('sourceFilter');
const sourceFilterMobile = document.getElementById('sourceFilterMobile');
const sourceStorageKey = 'dashboard-source-filter';
const SOURCE_LABELS = {
    both: 'All',
    claude_code: 'Claude Code',
    codex: 'Codex',
    copilot_chat: 'Copilot Chat',
    cursor: 'Cursor',
    lmstudio: 'LM Studio',
};
const SOURCE_ACCENTS = {
    claude_code: '#e67e22',
    codex: '#a855f7',
    copilot_chat: '#06b6d4',
    cursor: '#3b82f6',
    lmstudio: '#22c55e',
};

function formatSourceLabel(key) {
    return SOURCE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function getSourceDisplayName(key, short = false) {
    if (short) {
        if (key === 'claude_code') return 'Claude';
        if (key === 'copilot_chat') return 'Copilot';
    }
    return formatSourceLabel(key);
}

// === Formatting helpers ===

function formatNumber(value) {
    const n = Math.round(Number(value) || 0);
    return n.toLocaleString();
}

function formatCompact(value) {
    const n = Number(value) || 0;
    if (n < 1_000_000) return Math.round(n).toLocaleString();
    if (n < 1_000_000_000) {
        const m = n / 1_000_000;
        return (m >= 100 ? Math.round(m) : m.toFixed(1).replace(/\.0$/, '')) + 'M';
    }
    const b = n / 1_000_000_000;
    return (b >= 100 ? Math.round(b) : b.toFixed(1).replace(/\.0$/, '')) + 'B';
}

function formatHour12(hour) {
    const h = Number(hour) || 0;
    return `${h % 12 || 12}${h < 12 ? 'am' : 'pm'}`;
}

function formatDateRange(dateRange) {
    if (!dateRange || !dateRange.first || !dateRange.last) return '2025';
    const first = new Date(dateRange.first);
    const last = new Date(dateRange.last);
    const opts = { month: 'short', day: '2-digit', year: 'numeric' };
    return `${first.toLocaleDateString('en-US', opts)} – ${last.toLocaleDateString('en-US', opts)}`;
}

// === Player card ===

const CARD_IMAGES = {
    'The Architect': '/images/card_architect.png',
    'The Explorer': '/images/card_explorer.png',
    'The Commander': '/images/card_commander.png',
    'The Partner': '/images/card_partner.png',
};

const DONUT_CIRCUMFERENCE = 125.66;

function setDonut(id, valId, score) {
    const circle = document.getElementById(id);
    const valEl = document.getElementById(valId);
    const value = Math.round(score ?? 0);
    if (circle) {
        const offset = DONUT_CIRCUMFERENCE * (1 - Math.min(value, 100) / 100);
        circle.setAttribute('stroke-dashoffset', String(offset));
    }
    if (valEl) valEl.textContent = value;
}

function renderPlayerCard(persona, nlp, politeness, view) {
    const cardTop = document.getElementById('cardTop');
    const personaName = persona.name || 'The Explorer';
    if (cardTop) {
        cardTop.style.backgroundImage = `url('${CARD_IMAGES[personaName] || CARD_IMAGES['The Explorer']}')`;
    }

    const personaNameEl = document.getElementById('personaName');
    if (personaNameEl) personaNameEl.textContent = personaName;

    const personaDesc = document.getElementById('personaDescription');
    if (personaDesc) personaDesc.textContent = persona.description || 'Not enough data.';

    // 2×2 axes replace old radar donuts
    setDonut('donutDetail', 'valDetail', persona.detail_score);
    setDonut('donutStyle', 'valStyle', persona.style_score);

    const hitlScore = nlp?.hitl_score?.avg_score;
    const vibeScore = nlp?.vibe_coder_index?.avg_score;
    const politeScore = nlp?.politeness?.avg_score;

    const el = (id) => document.getElementById(id);
    if (el('cardHitl')) el('cardHitl').textContent = hitlScore != null ? Math.round(hitlScore) : '--';
    if (el('cardVibe')) el('cardVibe').textContent = vibeScore != null ? Math.round(vibeScore) : '--';
    if (el('cardPolite')) el('cardPolite').textContent = politeScore != null ? Math.round(politeScore) : '--';
    if (el('cardHitlBar')) el('cardHitlBar').style.width = `${Math.min(hitlScore ?? 0, 100)}%`;
    if (el('cardVibeBar')) el('cardVibeBar').style.width = `${Math.min(vibeScore ?? 0, 100)}%`;
    if (el('cardPoliteBar')) el('cardPoliteBar').style.width = `${Math.min(politeScore, 100)}%`;

    const serial = el('cardSerial');
    if (serial) {
        const hash = String(Math.abs((persona.name || '').split('').reduce((a, c) => ((a << 5) - a) + c.charCodeAt(0), 0)) % 10000).padStart(4, '0');
        serial.textContent = `#${hash}`;
    }

}

// === Heatmap ===

function renderHeatmap(heatmapData) {
    const data = Array.isArray(heatmapData) ? heatmapData : [];
    const heatmap = document.getElementById('heatmapGrid');
    if (!heatmap) return;
    // Apply platform color to heatmap cells
    const heatmapPanel = heatmap.closest('.heatmap-panel');
    if (heatmapPanel) heatmapPanel.style.setProperty('--heatmap-color', resolveAccent());

    // Ensure tooltip element exists
    let tooltip = heatmapPanel?.querySelector('.heatmap-tooltip');
    if (!tooltip && heatmapPanel) {
        tooltip = document.createElement('div');
        tooltip.className = 'heatmap-tooltip';
        heatmapPanel.appendChild(tooltip);
    }
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    // Empty corner + hour headers
    heatmap.innerHTML = '';
    const corner = document.createElement('div');
    heatmap.appendChild(corner);
    for (let h = 0; h < 24; h++) {
        const hourEl = document.createElement('div');
        hourEl.className = 'heatmap-hour';
        hourEl.textContent = (h % 3 === 0) ? String(h) : '';
        heatmap.appendChild(hourEl);
    }

    const maxVal = Math.max(1, ...data.flat());
    days.forEach((day, dayIndex) => {
        const label = document.createElement('div');
        label.className = 'heatmap-label';
        label.textContent = day;
        heatmap.appendChild(label);

        (data[dayIndex] || []).forEach((value, hour) => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            if (value > 0) {
                const intensity = Math.ceil((value / maxVal) * 5);
                cell.classList.add('l' + Math.min(intensity, 5));
            }
            cell.setAttribute('aria-label', `${day} ${formatHour12(hour)}: ${value} prompt${value !== 1 ? 's' : ''}`);
            cell.dataset.day = day;
            cell.dataset.hour = hour;
            cell.dataset.count = value;
            heatmap.appendChild(cell);
        });
    });

    // Tooltip via event delegation
    if (tooltip) {
        heatmap.onmouseover = (e) => {
            const cell = e.target.closest('.heatmap-cell');
            if (!cell) { tooltip.style.opacity = '0'; return; }
            const { day: d, hour: h, count: c } = cell.dataset;
            tooltip.textContent = `${d} ${formatHour12(h)} — ${c} prompt${c !== '1' ? 's' : ''}`;
            const rect = cell.getBoundingClientRect();
            const panelRect = heatmapPanel.getBoundingClientRect();
            let left = rect.left - panelRect.left + rect.width / 2;
            left = Math.max(60, Math.min(left, panelRect.width - 60));
            tooltip.style.left = left + 'px';
            tooltip.style.top = (rect.top - panelRect.top - 30) + 'px';
            tooltip.style.opacity = '1';
        };
        heatmap.onmouseleave = () => { tooltip.style.opacity = '0'; };
    }
}

// === Trend Chart (ApexCharts) ===

let trendData = {};
let activeTrendMetric = 'hitl';

const TREND_METRIC_CONFIG = {
    hitl:      { key: 'hitl_score',          label: 'Human in the Loop', suffix: '/100', desc: 'How actively you steer AI output. Higher = more hands-on review and iteration.' },
    vibe:      { key: 'vibe_coder_index',    label: 'Vibe Coder Index',  suffix: '/100', desc: 'Engineer vs. vibe coder spectrum. Higher = more structured, spec-driven prompting.' },
    polite:    { key: 'politeness',          label: 'Politeness',        suffix: '/100', desc: 'How courteous and collaborative your tone is. Higher = warmer, more appreciative prompting style.' },
    activity:  { key: '_prompts',            label: 'Activity',          suffix: '/wk', desc: 'Prompts per week.' },
};

function extractTrendPoints(weekly, metricKey) {
    return weekly.map((w) => {
        if (metricKey === '_prompts') return w.prompts ?? null;
        if (w.nlp && w.nlp[metricKey] != null) return Math.round(w.nlp[metricKey]);
        return null;
    });
}

function resolveAccent() {
    const platColor = SOURCE_ACCENTS[activeSourceKey];
    if (platColor) return platColor;
    return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#5c3d2e';
}

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

let trendChartInstance = null;

function initTrendChart(view) {
    const trends = view?.trends || {};
    const weekly = trends?.weekly_rollups || [];
    if (weekly.length < 2) return;

    const weeklyByPlatform = trends?.weekly_by_platform || {};
    const isAllView = Object.keys(weeklyByPlatform).length > 0;

    const chartEl = document.getElementById('trendChart');
    const valEl = document.getElementById('trendVal');
    const labelEl = document.getElementById('trendLabel');

    if (!chartEl) return;

    // Build week date labels
    const weekDates = weekly.map((w) => {
        const d = new Date(w.week_start || w.date);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });

    // Lifetime averages for headline display
    const nlp = view.nlp || {};
    const avgPromptsPerWeek = weekly.length > 0
        ? Math.round(weekly.reduce((s, w) => s + w.prompts, 0) / weekly.length)
        : null;
    const lifetimeAvg = {
        hitl: nlp.hitl_score?.avg_score,
        vibe: nlp.vibe_coder_index?.avg_score,
        polite: nlp.politeness?.avg_score,
        activity: avgPromptsPerWeek,
    };

    // Build trendData structure
    trendData = {};
    for (const [key, config] of Object.entries(TREND_METRIC_CONFIG)) {
        const points = extractTrendPoints(weekly, config.key);
        const hasData = points.some((p) => p != null);
        if (hasData) {
            const entry = {
                points: points.map((p) => p ?? 0),
                label: config.label,
                suffix: config.suffix,
                lifetime: lifetimeAvg[key],
            };

            if (isAllView) {
                const weekKeys = weekly.map((w) => w.week_start);
                entry.platforms = {};
                for (const [plat, platWeekly] of Object.entries(weeklyByPlatform)) {
                    const platByWeek = new Map(platWeekly.map((w) => [w.week_start, w]));
                    const platPoints = weekKeys.map((wk) => {
                        const pw = platByWeek.get(wk);
                        if (!pw) return 0;
                        if (config.key === '_prompts') return pw.prompts ?? 0;
                        if (pw.nlp && pw.nlp[config.key] != null) return Math.round(pw.nlp[config.key]);
                        return 0;
                    });
                    if (platPoints.some((p) => p != null && p !== 0)) {
                        entry.platforms[plat] = platPoints;
                    }
                }
            }

            trendData[key] = entry;
        }
    }

    if (!trendData[activeTrendMetric]) {
        activeTrendMetric = Object.keys(trendData)[0] || 'hitl';
    }

    function buildSeriesAndColors(metricKey) {
        const d = trendData[metricKey];
        if (!d) return { series: [], colors: [] };

        const hasPlatforms = d.platforms && Object.keys(d.platforms).length > 1;

        if (hasPlatforms) {
            const series = [];
            const colors = [];
            for (const [plat, platPts] of Object.entries(d.platforms)) {
                series.push({
                    name: SOURCE_LABELS[plat] || plat,
                    data: platPts.map((v) => v ?? 0),
                });
                colors.push(SOURCE_ACCENTS[plat] || '#666666');
            }
            return { series, colors, showLegend: true };
        } else {
            return {
                series: [{ name: d.label, data: d.points }],
                colors: [resolveAccent()],
                showLegend: false,
            };
        }
    }

    const isDark = document.documentElement.classList.contains('dark');
    const mutedColor = cssVar('--muted') || '#888';
    const initial = buildSeriesAndColors(activeTrendMetric);
    const initialSuffix = trendData[activeTrendMetric]?.suffix || '';

    const options = {
        chart: {
            type: 'area',
            height: 220,
            fontFamily: "'DM Sans', system-ui, sans-serif",
            toolbar: { show: false },
            zoom: { enabled: false },
            background: 'transparent',
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 400,
            },
        },
        stroke: {
            curve: 'smooth',
            width: 2.5,
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.35,
                opacityTo: 0.02,
                stops: [0, 95],
            },
        },
        colors: initial.colors,
        series: initial.series,
        xaxis: {
            categories: weekDates,
            labels: {
                style: { colors: mutedColor, fontSize: '11px' },
                rotate: 0,
                hideOverlappingLabels: true,
            },
            tickAmount: 7,
            axisBorder: { show: false },
            axisTicks: { show: false },
        },
        yaxis: {
            show: false,
            min: 0,
        },
        grid: {
            show: false,
            padding: { left: 16, right: 16, top: -8, bottom: 0 },
        },
        legend: { show: false },
        tooltip: {
            shared: true,
            intersect: false,
            theme: isDark ? 'dark' : 'light',
            y: {
                formatter: (val) => val != null ? Math.round(val) + initialSuffix : '--',
            },
            style: { fontSize: '12px', fontFamily: "'JetBrains Mono', monospace" },
        },
        dataLabels: { enabled: false },
    };

    // Destroy existing instance before re-creating
    if (trendChartInstance) {
        trendChartInstance.destroy();
        trendChartInstance = null;
    }

    trendChartInstance = new ApexCharts(chartEl, options);
    trendChartInstance.render();

    function setMetric(key) {
        if (!trendData[key]) return;
        activeTrendMetric = key;
        const d = trendData[key];
        const headlineRaw = d.lifetime != null ? Math.round(d.lifetime) : d.points[d.points.length - 1];
        valEl.textContent = (key === 'activity' ? formatCompact(headlineRaw) : headlineRaw) + d.suffix;
        labelEl.textContent = `${d.label} \u00B7 ${weekly.length}-week trend`;

        // Set headline color
        const trendPanel = valEl.closest('.trend-panel');
        if (trendPanel) {
            trendPanel.style.setProperty('--trend-color', resolveAccent());
        }

        document.querySelectorAll('.metric-tab').forEach((t) =>
            t.classList.toggle('active', t.dataset.metric === key)
        );

        const { series, colors, showLegend } = buildSeriesAndColors(key);
        const suffix = d.suffix || '';

        // Update inline legend
        const legendEl = document.getElementById('trendLegend');
        if (legendEl) {
            legendEl.innerHTML = '';
            if (showLegend) {
                for (const s of series) {
                    const plat = Object.entries(SOURCE_LABELS).find(([, v]) => v === s.name)?.[0];
                    const color = plat ? SOURCE_ACCENTS[plat] : '#666';
                    const pill = document.createElement('span');
                    pill.className = 'trend-legend-pill';
                    pill.innerHTML = `<span class="trend-legend-dot" style="background:${color}"></span>${s.name}`;
                    legendEl.appendChild(pill);
                }
                legendEl.style.display = '';
            } else {
                legendEl.style.display = 'none';
            }
        }

        trendChartInstance.updateOptions({
            colors,
            tooltip: {
                y: {
                    formatter: (val) => val != null ? Math.round(val) + suffix : '--',
                },
            },
        }, false, false);
        trendChartInstance.updateSeries(series);
    }

    document.querySelectorAll('.metric-tab').forEach((tab) => {
        tab.addEventListener('click', () => setMetric(tab.dataset.metric));
    });

    setMetric(activeTrendMetric);
}

// === 3D Tilt (desktop hover) ===

function initCardTilt() {
    const card = document.getElementById('playerCard');
    const dock = card?.closest('.card-dock');
    if (!card || !dock) return;
    const MAX_TILT = 12;

    dock.addEventListener('mousemove', (e) => {
        const r = dock.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width;
        const y = (e.clientY - r.top) / r.height;
        card.style.transform = `rotateY(${(x - 0.5) * MAX_TILT}deg) rotateX(${(0.5 - y) * MAX_TILT}deg)`;
    });
    dock.addEventListener('mouseleave', () => { card.style.transform = ''; });
}

// === Share / Download Card ===

async function captureCard() {
    if (typeof html2canvas === 'undefined') return null;
    const element = document.getElementById('playerCard');
    if (!element) return null;
    const canvas = await html2canvas(element, { backgroundColor: null, scale: 2, useCORS: true });
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), 'image/png'));
}

function showShareToast(msg) {
    const toast = document.getElementById('shareToast');
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2500);
}

function getShareText() {
    const persona = document.getElementById('personaName')?.textContent || '';
    return `My AI prompting persona: ${persona}. See yours at howiprompt.com`;
}

async function handleShareAction(action) {
    const dropdown = document.getElementById('shareDropdown');
    dropdown?.classList.remove('open');

    try {
        if (action === 'download') {
            const blob = await captureCard();
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'howiprompt-card.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showShareToast('Card downloaded!');
        } else if (action === 'copy') {
            const blob = await captureCard();
            if (!blob) return;
            await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
            showShareToast('Copied to clipboard!');
        } else if (action === 'twitter') {
            const text = encodeURIComponent(getShareText());
            window.open(`https://twitter.com/intent/tweet?text=${text}`, '_blank');
        } else if (action === 'linkedin') {
            const text = encodeURIComponent(getShareText());
            window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent('https://howiprompt.com')}&summary=${text}`, '_blank');
        }
    } catch (err) {
        console.warn('Share action failed:', err.message);
        showShareToast('Share failed — try downloading instead');
    }
}

// Toggle dropdown
document.getElementById('shareToggle')?.addEventListener('click', (e) => {
    e.stopPropagation();
    document.getElementById('shareDropdown')?.classList.toggle('open');
});

// Handle menu item clicks
document.getElementById('shareMenu')?.addEventListener('click', (e) => {
    const item = e.target.closest('[data-action]');
    if (item) handleShareAction(item.dataset.action);
});

// Close dropdown on outside click
document.addEventListener('click', () => {
    document.getElementById('shareDropdown')?.classList.remove('open');
});

// === Main renderView ===

function getView(sourceKey) {
    return sourceViews[sourceKey] || null;
}

function getAvailableSourceKeys() {
    return Object.keys(sourceViews)
        .filter((key) => key === 'both' || Boolean(getView(key)))
        .sort((a, b) => {
            if (a === 'both') return -1;
            if (b === 'both') return 1;
            return formatSourceLabel(a).localeCompare(formatSourceLabel(b));
        });
}

function renderView(sourceKey) {
    const view = getView(sourceKey);
    if (!view) return;
    activeSourceKey = sourceKey;
    activeView = view;

    const volume = view.volume || {};
    const temporal = view.temporal || {};
    const convo = view.conversation_depth || {};
    const politeness = view.politeness || {};
    const persona = view.persona || {};

    const el = (id) => document.getElementById(id);

    // Date range
    const dateRange = el('dateRange');
    if (dateRange) dateRange.textContent = formatDateRange(view.date_range);

    // Stat cards
    const promptsValue = el('promptsValue');
    if (promptsValue) promptsValue.textContent = formatNumber(volume.total_human);
    const promptsSubtitle = el('promptsSubtitle');
    if (promptsSubtitle) promptsSubtitle.textContent = `${formatNumber(volume.avg_words_per_prompt)} words avg`;

    const conversationsValue = el('conversationsValue');
    if (conversationsValue) conversationsValue.textContent = formatNumber(volume.total_conversations);
    const conversationsSubtitle = el('conversationsSubtitle');
    if (conversationsSubtitle) conversationsSubtitle.textContent = `${convo.avg_turns ?? 0} turns avg`;

    const wordsTypedValue = el('wordsTypedValue');
    if (wordsTypedValue) wordsTypedValue.textContent = formatCompact(volume.total_words_human);
    const wordsTypedSubtitle = el('wordsTypedSubtitle');
    if (wordsTypedSubtitle) wordsTypedSubtitle.textContent = `${volume.avg_words_per_prompt ?? 0} words avg`;

    const nightOwlValue = el('nightOwlValue');
    if (nightOwlValue) nightOwlValue.textContent = `${temporal.night_owl_pct ?? 0}%`;

    // Heatmap meta
    const peakHourValue = el('peakHourValue');
    if (peakHourValue) peakHourValue.textContent = formatHour12(temporal.peak_hour);
    const peakDayValue = el('peakDayValue');
    if (peakDayValue) peakDayValue.textContent = temporal.peak_day || 'N/A';

    // Player card
    renderPlayerCard(persona, view.nlp || {}, politeness, view);

    // Heatmap
    renderHeatmap(temporal.heatmap);

    // Trend chart
    initTrendChart(view);
}

// === Source filter ===

function syncSourceSelectors(value) {
    if (sourceFilter) sourceFilter.value = value;
    if (sourceFilterMobile) sourceFilterMobile.value = value;
}

function initSourceFilter() {
    if (!sourceFilter) return;

    const available = getAvailableSourceKeys();
    [sourceFilter, sourceFilterMobile].filter(Boolean).forEach((select) => {
        select.innerHTML = '';
        for (const key of available) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = formatSourceLabel(key);
            select.appendChild(opt);
        }
    });

    let selected = localStorage.getItem(sourceStorageKey) || defaultSource;
    if (selected === 'claude' && available.includes('claude_code')) selected = 'claude_code';
    if (!available.includes(selected)) selected = available[0] || 'both';

    syncSourceSelectors(selected);
    renderView(selected);

    const handleChange = (event) => {
        const next = event.target.value;
        if (!available.includes(next)) return;
        syncSourceSelectors(next);
        localStorage.setItem(sourceStorageKey, next);
        renderView(next);
    };

    sourceFilter.addEventListener('change', handleChange);
    sourceFilterMobile?.addEventListener('change', handleChange);
}

// === Methodology modal ===

let methodologyOpener = null;

function openMethodology() {
    const modal = document.getElementById('methodologyModal');
    if (modal) {
        methodologyOpener = document.activeElement;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        const closeBtn = modal.querySelector('.modal-close');
        if (closeBtn) closeBtn.focus();
    }
}

function closeMethodology() {
    const modal = document.getElementById('methodologyModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
        if (methodologyOpener) { methodologyOpener.focus(); methodologyOpener = null; }
    }
}

window.openMethodology = openMethodology;
window.closeMethodology = closeMethodology;

// === Settings modal ===

let settingsOpener = null;
let detectedBackendCache = null;
let detectedBackendFetch = null;

function getCachedDetectedBackends() {
    if (Array.isArray(wizardBackendData) && wizardBackendData.length > 0) return wizardBackendData;
    if (Array.isArray(detectedBackendCache) && detectedBackendCache.length > 0) return detectedBackendCache;
    return [];
}

function setDetectedBackends(backends) {
    const normalized = Array.isArray(backends) ? backends : [];
    detectedBackendCache = normalized;
    if (normalized.length > 0) wizardBackendData = normalized;
    return normalized;
}

function summarizeDetectionChanges(previousBackends = [], nextBackends = [], knownBackendIds = []) {
    const previousById = new Map((previousBackends || []).map((backend) => [backend.id, backend]));
    const knownIds = new Set(knownBackendIds);
    const badges = {};

    for (const backend of nextBackends || []) {
        const previous = previousById.get(backend.id);
        if (!previous) {
            if (!knownIds.has(backend.id)) badges[backend.id] = 'New';
            continue;
        }
        if (previous.status !== backend.status || previous.detected !== backend.detected) {
            badges[backend.id] = 'Updated';
        }
    }

    return badges;
}

async function fetchDetectedBackends(options = {}) {
    const { useCache = true, force = false } = options;

    if (!force) {
        const cached = getCachedDetectedBackends();
        if (useCache && cached.length > 0) return cached;
        if (detectedBackendFetch) return detectedBackendFetch;
    }

    detectedBackendFetch = fetch('/api/detect')
        .then((r) => r.json())
        .then((payload) => setDetectedBackends(payload.backends || []))
        .catch(() => getCachedDetectedBackends())
        .finally(() => {
            detectedBackendFetch = null;
        });

    return detectedBackendFetch;
}

function renderSettingsBackends(configRes = {}, detectedBackends = [], options = {}) {
    const { isRefreshing = false, badges = {} } = options;
    const container = document.getElementById('settingsBackends');
    if (!container) return;

    const detectedById = new Map((detectedBackends || []).map((backend) => [backend.id, backend]));
    const backendIds = Array.from(new Set([
        ...Object.keys(configRes.backends || {}),
        ...Array.from(detectedById.keys()),
    ]));

    if (backendIds.length === 0) {
        container.innerHTML = `<div class="wizard-loading">${isRefreshing ? 'Checking installed backends…' : 'No backend settings yet.'}</div>`;
        return;
    }

    container.innerHTML = backendIds.map(id => {
        const toggle = configRes.backends?.[id] || { enabled: false, exclusions: [] };
        const info = detectedById.get(id);
        const disabled = info ? (info.supported === false || info.status === 'coming_soon' || info.status === 'not_found') : false;
        let detail = isRefreshing ? 'Checking installation...' : 'Saved configuration';
        if (info?.status === 'available') detail = 'Ready to analyze';
        if (info?.status === 'coming_soon') detail = 'Detected, but analysis support is not shipped yet';
        if (info?.status === 'not_found') detail = 'Not installed';
        const badge = badges[id] ? `<span class="settings-backend-badge">${badges[id]}</span>` : '';
        return `<div class="settings-backend-row">
            <label class="wizard-toggle ${disabled ? 'is-disabled' : ''}">
                <input type="checkbox" data-backend="${id}" ${toggle.enabled !== false ? 'checked' : ''} ${disabled ? 'disabled' : ''}>
                <span class="settings-backend-title">${formatSourceLabel(id)}${badge}</span>
            </label>
            <div class="settings-backend-detail">${detail}</div>
        </div>`;
    }).join('');
}

async function openSettings() {
    const modal = document.getElementById('settingsModal');
    if (!modal) return;

    settingsOpener = document.activeElement;

    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    const configRes = await fetch('/api/config').then((r) => r.json()).catch(() => ({}));
    const initialDetected = getCachedDetectedBackends();
    renderSettingsBackends(configRes, initialDetected, { isRefreshing: true });

    // Show exclusions if Claude Code is configured
    const excSection = document.getElementById('settingsExclusionSection');
    if (excSection && configRes.backends?.claude_code) {
        excSection.style.display = 'block';
        const exclusions = configRes.backends.claude_code.exclusions ?? [];
        loadExclusionChips('settingsExclusionChips', exclusions);
    }

    fetchDetectedBackends({ force: true })
        .then((backends) => {
            const stillOpen = document.getElementById('settingsModal')?.classList.contains('active');
            if (!stillOpen) return;
            const knownBackendIds = [
                ...initialDetected.map((backend) => backend.id),
                ...Object.keys(configRes.backends || {}),
            ];
            const badges = summarizeDetectionChanges(initialDetected, backends, knownBackendIds);
            renderSettingsBackends(configRes, backends, { isRefreshing: false, badges });
        })
        .catch(() => {});
}

function closeSettings() {
    const modal = document.getElementById('settingsModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
        if (settingsOpener) { settingsOpener.focus(); settingsOpener = null; }
    }
}

async function saveSettings() {
    const checked = document.querySelectorAll('#settingsBackends input[type=checkbox]:checked');
    if (checked.length === 0) {
        alert('Please enable at least one backend to analyze.');
        return;
    }

    const toggles = {};
    document.querySelectorAll('#settingsBackends input[type=checkbox]').forEach(cb => {
        toggles[cb.dataset.backend] = { enabled: cb.checked, exclusions: [] };
    });
    if (toggles.claude_code) {
        toggles.claude_code.exclusions = getExclusionPaths('settingsExclusionChips');
    }

    await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backends: toggles }),
    });

    closeSettings();
    handleRefresh();
}

window.openSettings = openSettings;
window.closeSettings = closeSettings;

document.getElementById('settingsAddExclusion')?.addEventListener('click', () => pickAndAddExclusion('settingsExclusionChips'));
document.getElementById('settingsSave')?.addEventListener('click', saveSettings);
document.getElementById('settingsReset')?.addEventListener('click', () => {
    document.getElementById('resetConfirmModal')?.classList.add('active');
});
document.getElementById('resetCancel')?.addEventListener('click', () => {
    document.getElementById('resetConfirmModal')?.classList.remove('active');
});
document.getElementById('resetConfirm')?.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    window.location.reload();
});
document.getElementById('settingsModal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeSettings();
});

document.getElementById('methodologyModal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeMethodology();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeMethodology(); closeSettings(); }
    // Focus trap for methodology modal
    const modal = document.getElementById('methodologyModal');
    if (e.key === 'Tab' && modal?.classList.contains('active')) {
        const focusable = modal.querySelectorAll('button, [href], [tabindex]:not([tabindex="-1"])');
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
            e.preventDefault(); last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault(); first.focus();
        }
    }
});

// === Fix local links ===

function fixLocalLinks() {
    const isLocal = location.protocol === 'file:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (isLocal) {
        document.querySelectorAll('a[href="/wrapped"]').forEach((link) => {
            const currentPath = location.pathname;
            const basePath = currentPath.substring(0, currentPath.lastIndexOf('/'));
            link.href = basePath + '/wrapped/index.html';
        });
    }
}

// === Apply branding ===

function applyBranding(metricsData) {
    branding = metricsData.branding || {};
    const githubRepo = branding.github_repo || 'https://github.com/eeshansrivastava89/howiprompt';
    const footerGithubLink = document.getElementById('footerGithubLink');
    if (footerGithubLink) footerGithubLink.href = githubRepo;
}

// === Refresh ===

async function handleRefresh() {
    const btn = document.getElementById('refreshBtn');
    if (!btn || btn.classList.contains('refreshing')) return;

    const modal = document.getElementById('refreshModal');
    const log = document.getElementById('refreshLog');
    const bar = document.getElementById('refreshProgressBar');
    const resultEl = document.getElementById('refreshModalResult');
    const closeBtn = document.getElementById('refreshModalClose');

    // Show modal with starter message
    btn.classList.add('refreshing');
    if (modal) {
        if (log) log.innerHTML = '<div class="wizard-log-entry">Starting pipeline...</div>';
        if (bar) bar.style.width = '0';
        resultEl.style.display = 'none';
        closeBtn.style.display = 'none';
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    const showResult = (html) => {
        if (resultEl) {
            resultEl.innerHTML = html;
            resultEl.style.display = 'block';
            closeBtn.style.display = 'inline-block';
            closeBtn.focus();
        }
        btn.classList.remove('refreshing');
        const label = btn.querySelector('.refresh-label');
        if (label) label.textContent = 'Refresh';
    };

    // Reuse same SSE log pattern as wizard
    const stages = ['sync', 'parse', 'insert', 'nlp', 'embedding', 'classifiers', 'metrics'];
    let maxStageIdx = 0;
    let refreshDone = false;
    const evtSource = new EventSource('/api/pipeline/stream');

    evtSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        if (log) {
            const entry = document.createElement('div');
            entry.className = 'wizard-log-entry';
            entry.textContent = `${data.stage}: ${data.detail}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        const idx = stages.indexOf(data.stage);
        if (idx >= 0 && idx > maxStageIdx) maxStageIdx = idx;
        if (bar) bar.style.width = `${((maxStageIdx + 1) / stages.length) * 100}%`;
    });

    evtSource.addEventListener('complete', (e) => {
        refreshDone = true;
        evtSource.close();
        const data = JSON.parse(e.data);

        if (bar) bar.style.width = '100%';
        if (log) {
            const entry = document.createElement('div');
            entry.className = 'wizard-log-entry done';
            entry.textContent = `Done! ${data.stats.totalMessages.toLocaleString()} messages analyzed.`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }

        if (data.metrics) {
            sourceViews = data.metrics.source_views || { both: data.metrics };
            if (!sourceViews.claude_code && sourceViews.claude) sourceViews.claude_code = sourceViews.claude;
            sourceViews.both = sourceViews.both || data.metrics;
            // Fall back to 'both' if the active source was disabled
            if (!sourceViews[activeSourceKey]) activeSourceKey = 'both';
            initSourceFilter();
            renderView(activeSourceKey);
        }

        const lines = [];
        const stats = data.stats;
        if (stats.newMessages > 0) {
            lines.push(`<span class="result-num">${stats.newMessages}</span> new message${stats.newMessages === 1 ? '' : 's'} synced`);
        } else {
            lines.push('Already up to date');
        }
        lines.push(`<span class="result-num">${formatCompact(stats.totalMessages)}</span> total messages`);
        if (stats.embedded > 0) {
            lines.push(`<span class="result-num">${stats.embedded}</span> embeddings computed`);
        }
        showResult(lines.join('<br>'));
    });

    evtSource.addEventListener('pipeline_error', (e) => {
        refreshDone = true;
        evtSource.close();
        const data = JSON.parse(e.data);
        showResult(`Refresh failed: ${data.message}`);
    });

    evtSource.onerror = () => {
        if (!refreshDone) {
            evtSource.close();
            showResult('Refresh failed — is the server running?');
        }
    };
}

document.getElementById('refreshModalClose')?.addEventListener('click', () => {
    const modal = document.getElementById('refreshModal');
    if (modal) { modal.classList.remove('active'); document.body.style.overflow = ''; }
});

document.getElementById('refreshBtn')?.addEventListener('click', handleRefresh);

// === Tab switching ===

function initTabs() {
    const tabs = document.querySelectorAll('.tab-bar .tab');
    const panels = document.querySelectorAll('.tab-panel');

    function switchTab(tabName) {
        tabs.forEach((t) => {
            const isActive = t.dataset.tab === tabName;
            t.classList.toggle('active', isActive);
            t.setAttribute('aria-selected', String(isActive));
        });
        panels.forEach((p) => p.classList.toggle('active', p.id === `tab-${tabName}`));
        window.location.hash = tabName;

        if (tabName === 'leaderboard') fetchLeaderboard();
    }

    tabs.forEach((tab) => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    const hash = window.location.hash.replace('#', '');
    if (hash && document.getElementById(`tab-${hash}`)) switchTab(hash);

    window.addEventListener('hashchange', () => {
        const h = window.location.hash.replace('#', '');
        if (h && document.getElementById(`tab-${h}`)) switchTab(h);
    });
}

// === Leaderboard (Supabase PostgREST) ===

const SUPABASE_URL = 'https://nazioidbiydxduonenmb.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5hemlvaWRiaXlkeGR1b25lbm1iIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2NDEyMjQsImV4cCI6MjA3NjIxNzIyNH0.PjEzSI8wq74RCQpSkh7j4zhh_5nXc2nYX0M5vCjLEro';
const LB_TABLE = '/rest/v1/howiprompt_leaderboard';

function supaHeaders(includeContent = true) {
    const h = { apikey: SUPABASE_ANON_KEY, Authorization: `Bearer ${SUPABASE_ANON_KEY}` };
    if (includeContent) h['Content-Type'] = 'application/json';
    return h;
}

let leaderboardData = [];
let leaderboardLoaded = false;

async function fetchLeaderboard() {
    if (leaderboardLoaded) return;
    try {
        const res = await fetch(`${SUPABASE_URL}${LB_TABLE}?select=*&order=hitl_score.desc`, {
            headers: supaHeaders(false),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        leaderboardData = await res.json();
        leaderboardLoaded = true;
        renderLeaderboard();
    } catch {
        leaderboardData = [];
        renderLeaderboard();
    }
}

function renderLeaderboard() {
    const table = document.getElementById('leaderboardTable');
    const empty = document.getElementById('leaderboardEmpty');
    const body = document.getElementById('leaderboardBody');
    if (!table || !empty || !body) return;

    if (leaderboardData.length === 0) {
        table.style.display = 'none';
        empty.style.display = '';
        return;
    }

    empty.style.display = 'none';
    table.style.display = '';

    const sortKey = document.getElementById('leaderboardSort')?.value || 'hitl_score';
    const sorted = sortLeaderboardEntries(leaderboardData, sortKey);
    const clientId = localStorage.getItem(CLIENT_ID_STORAGE_KEY);

    body.innerHTML = sorted.map((entry, i) => {
        const isYou = clientId && entry.fingerprint === clientId;
        return `
        <tr${isYou ? ' class="is-you"' : ''}>
            <td class="rank-col">${i + 1}</td>
            <td><strong>${escapeHtml(entry.display_name || 'Anonymous')}</strong>${isYou ? ' <span style="font-size:11px;color:var(--accent)">(you)</span>' : ''}</td>
            <td>${entry.hitl_score ?? '--'}</td>
            <td>${entry.vibe_index ?? '--'}</td>
            <td>${entry.politeness ?? '--'}</td>
            <td>${(entry.total_prompts ?? 0).toLocaleString()}</td>
            <td>${(entry.total_conversations ?? 0).toLocaleString()}</td>
            <td>${escapeHtml(entry.persona || '--')}</td>
            <td>${isYou ? '<button class="delete-entry-btn" onclick="deleteMySubmission()">✕</button>' : ''}</td>
        </tr>`;
    }).join('');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

document.getElementById('leaderboardSort')?.addEventListener('change', renderLeaderboard);

// === Username + Fingerprint ===

function getOrCreateUsername() {
    let name = localStorage.getItem(USERNAME_STORAGE_KEY);
    if (!name && typeof window.generateRandomUsername === 'function') {
        name = window.generateRandomUsername();
        localStorage.setItem(USERNAME_STORAGE_KEY, name);
    }
    return name || 'Anonymous';
}

// === Submit to leaderboard ===

function getSubmissionPayload() {
    return buildSubmissionPayload(activeView, activeSourceKey);
}

function showSubmitModal() {
    const modal = document.getElementById('submitModal');
    const preview = document.getElementById('submitPreview');
    const nameInput = document.getElementById('displayName');
    if (!modal || !preview) return;

    const payload = getSubmissionPayload();
    if (!payload) return;

    // Pre-fill username
    if (nameInput) nameInput.value = getOrCreateUsername();

    const labels = {
        hitl_score: 'Human in the Loop', vibe_index: 'Vibe Index', politeness: 'Politeness',
        total_prompts: 'Prompts', total_conversations: 'Conversations', persona: 'Persona', platform: 'Platform',
    };
    preview.innerHTML = Object.entries(payload)
        .map(([k, v]) => `<div style="display:flex;justify-content:space-between;padding:2px 0;"><span style="color:var(--muted)">${labels[k] || k}</span><span>${v}</span></div>`)
        .join('');

    modal.classList.add('active');
    if (nameInput) nameInput.focus();
}

function hideSubmitModal() {
    document.getElementById('submitModal')?.classList.remove('active');
}

async function submitToLeaderboard() {
    const payload = getSubmissionPayload();
    const nameInput = document.getElementById('displayName');
    if (!payload || !nameInput) return;

    const displayName = nameInput.value.trim().slice(0, 30);
    if (!displayName) {
        nameInput.style.borderColor = 'red';
        nameInput.focus();
        return;
    }

    const clientId = createStableClientId(localStorage);
    if (!clientId) return;

    payload.display_name = displayName;
    payload.fingerprint = clientId;
    payload.updated_at = new Date().toISOString();

    localStorage.setItem(USERNAME_STORAGE_KEY, displayName);
    hideSubmitModal();

    const toast = document.getElementById('leaderboardToast');
    try {
        const res = await fetch(`${SUPABASE_URL}${LB_TABLE}`, {
            method: 'POST',
            headers: { ...supaHeaders(true), 'Prefer': 'resolution=merge-duplicates' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        leaderboardLoaded = false;
        await fetchLeaderboard();

        // Find rank
        const sortKey = document.getElementById('leaderboardSort')?.value || 'hitl_score';
        const rank = findEntryRank(leaderboardData, sortKey, clientId);

        if (toast) {
            toast.textContent = rank > 0 ? `Submitted! You're ranked #${rank}` : 'Scores submitted!';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }
    } catch {
        if (toast) {
            toast.textContent = 'Submit failed — try again later.';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }
    }
}

async function deleteMySubmission() {
    const clientId = localStorage.getItem(CLIENT_ID_STORAGE_KEY);
    if (!clientId) return;
    const toast = document.getElementById('leaderboardToast');
    try {
        const res = await fetch(`${SUPABASE_URL}${LB_TABLE}?fingerprint=eq.${clientId}`, {
            method: 'DELETE',
            headers: supaHeaders(false),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        leaderboardLoaded = false;
        fetchLeaderboard();
        if (toast) {
            toast.textContent = 'Submission deleted.';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2500);
        }
    } catch {
        if (toast) {
            toast.textContent = 'Delete failed — try again later.';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }
    }
}

window.deleteMySubmission = deleteMySubmission;

document.getElementById('submitLeaderboard')?.addEventListener('click', showSubmitModal);
document.getElementById('cancelSubmit')?.addEventListener('click', hideSubmitModal);
document.getElementById('confirmSubmit')?.addEventListener('click', submitToLeaderboard);
document.getElementById('submitModal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) hideSubmitModal();
});

// === Setup Wizard ===

let wizardBackendData = [];
const PIPELINE_STAGE_ORDER = [
    { key: 'boot', label: 'Prepare' },
    { key: 'sync', label: 'Sync Sources' },
    { key: 'parse', label: 'Parse Chats' },
    { key: 'insert', label: 'Store Messages' },
    { key: 'nlp', label: 'Language Scores' },
    { key: 'embedding', label: 'Embeddings' },
    { key: 'classifiers', label: 'Behavior Scores' },
    { key: 'metrics', label: 'Build Dashboard' },
];
const PIPELINE_STAGE_INDEX = Object.fromEntries(PIPELINE_STAGE_ORDER.map((stage, index) => [stage.key, index]));

function createWizardPipelineTracker() {
    const logEl = document.getElementById('wizardLog');
    const barEl = document.getElementById('wizardProgressBar');
    const labelEl = document.getElementById('wizardProgressLabel');
    const pctEl = document.getElementById('wizardProgressPct');
    const stageListEl = document.getElementById('wizardStageList');
    const stageState = Object.fromEntries(PIPELINE_STAGE_ORDER.map(({ key }, index) => [key, {
        key,
        label: PIPELINE_STAGE_ORDER[index].label,
        detail: 'Waiting...',
        progress: 0,
        status: 'pending',
    }]));
    let activeStage = 'boot';
    let trickleTimer = null;
    let completed = false;

    function render() {
        const overall = PIPELINE_STAGE_ORDER.reduce((sum, { key }) => sum + (stageState[key].progress || 0), 0) / PIPELINE_STAGE_ORDER.length;
        if (barEl) barEl.style.width = `${Math.round(overall)}%`;
        if (labelEl) {
            const active = stageState[activeStage] || stageState.boot;
            labelEl.textContent = active?.detail || 'Preparing analysis...';
        }
        if (pctEl) pctEl.textContent = `${Math.round(overall)}%`;
        if (stageListEl) {
            stageListEl.innerHTML = PIPELINE_STAGE_ORDER.map(({ key, label }) => {
                const stage = stageState[key];
                return `
                    <div class="wizard-stage-row">
                        <div class="wizard-stage-name">${label}</div>
                        <div class="wizard-stage-track is-${stage.status}">
                            <div class="wizard-stage-fill" style="width:${stage.progress}%"></div>
                            <span class="wizard-stage-detail">${stage.detail}</span>
                        </div>
                        <div class="wizard-stage-value">${Math.round(stage.progress)}%</div>
                    </div>
                `;
            }).join('');
        }
    }

    function stopTrickle() {
        if (trickleTimer) {
            clearInterval(trickleTimer);
            trickleTimer = null;
        }
    }

    function startTrickle(stageKey) {
        stopTrickle();
        trickleTimer = window.setInterval(() => {
            if (completed || activeStage !== stageKey) return;
            const stage = stageState[stageKey];
            const ceiling = stageKey === 'embedding' || stageKey === 'classifiers' ? 95 : 88;
            if (stage.progress >= ceiling) return;
            stage.progress = Math.min(ceiling, stage.progress + (stage.progress < 30 ? 3 : 1));
            render();
        }, 450);
    }

    function appendLog(detail, status = 'normal') {
        if (!logEl) return;
        const entry = document.createElement('div');
        entry.className = `wizard-log-entry${status === 'done' ? ' done' : status === 'error' ? ' error' : ''}`;
        entry.textContent = detail;
        logEl.appendChild(entry);
        logEl.scrollTop = logEl.scrollHeight;
    }

    function begin() {
        if (logEl) logEl.innerHTML = '';
        for (const { key } of PIPELINE_STAGE_ORDER) {
            stageState[key].detail = 'Waiting...';
            stageState[key].progress = 0;
            stageState[key].status = 'pending';
        }
        activeStage = 'boot';
        stageState.boot.detail = 'Connecting to local pipeline...';
        stageState.boot.progress = 5;
        stageState.boot.status = 'active';
        completed = false;
        appendLog('Starting pipeline...');
        startTrickle('boot');
        render();
    }

    function advance(stageKey, detail, progress) {
        const normalizedKey = PIPELINE_STAGE_INDEX[stageKey] != null ? stageKey : 'boot';
        const nextIndex = PIPELINE_STAGE_INDEX[normalizedKey];
        const prevIndex = PIPELINE_STAGE_INDEX[activeStage] ?? 0;

        if (nextIndex > prevIndex) {
            for (let i = prevIndex; i < nextIndex; i++) {
                const key = PIPELINE_STAGE_ORDER[i].key;
                stageState[key].status = 'done';
                stageState[key].progress = 100;
            }
        }

        activeStage = normalizedKey;
        const stage = stageState[normalizedKey];
        stage.status = 'active';
        stage.detail = detail;
        if (typeof progress === 'number' && Number.isFinite(progress)) {
            stage.progress = Math.max(stage.progress, Math.min(100, progress));
        } else {
            stage.progress = Math.max(stage.progress, 12);
        }
        appendLog(`${normalizedKey}: ${detail}`);
        startTrickle(normalizedKey);
        render();
    }

    function finish(totalMessages) {
        completed = true;
        stopTrickle();
        for (const { key } of PIPELINE_STAGE_ORDER) {
            stageState[key].status = 'done';
            stageState[key].progress = 100;
            if (stageState[key].detail === 'Waiting...') stageState[key].detail = 'Done';
        }
        activeStage = 'metrics';
        stageState.metrics.detail = 'Dashboard ready.';
        appendLog(`Done! ${Number(totalMessages || 0).toLocaleString()} messages analyzed.`, 'done');
        render();
    }

    function fail(message) {
        completed = true;
        stopTrickle();
        const stage = stageState[activeStage] || stageState.boot;
        stage.status = 'error';
        stage.detail = message;
        appendLog(`Error: ${message}`, 'error');
        render();
    }

    return { begin, advance, finish, fail };
}

async function initWizard() {
    const wizard = document.getElementById('setupWizard');
    if (!wizard) return false;

    // Check if metrics.json exists — if it does, no wizard needed
    let hasMetrics = false;
    try {
        const metricsRes = await fetch('./metrics.json');
        hasMetrics = metricsRes.ok;
    } catch { /* no metrics */ }

    if (hasMetrics) {
        // Ensure flag is set so future loads skip this check
        fetch('/api/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hasCompletedSetup: true }),
        }).catch(() => {});
        return false;
    }

    // Check config flag — only skip wizard if flag is set AND metrics exist
    let config;
    try {
        const res = await fetch('/api/config');
        if (res.ok) config = await res.json();
    } catch { /* server may not support /api/config yet */ }

    if (config?.hasCompletedSetup && hasMetrics) return false;

    wizard.classList.add('active');
    document.body.style.overflow = 'hidden';
    await wizardDetect();
    return true;
}

async function wizardDetect() {
    const container = document.getElementById('wizardBackends');
    if (!container) return;

    try {
        const res = await fetch('/api/detect');
        const { backends } = await res.json();
        setDetectedBackends(backends);

        container.innerHTML = backends.map(b => {
            let detail = '';
            if (b.status === 'available') {
                detail = 'Detected locally';
            } else if (b.status === 'coming_soon') {
                detail = 'Detected (coming soon)';
            } else {
                detail = 'Not installed';
            }
            return `
            <div class="wizard-backend ${b.detected ? 'detected' : ''}" data-id="${b.id}">
                <div class="wizard-backend-icon ${b.status}"></div>
                <div class="wizard-backend-info">
                    <div class="wizard-backend-name">${b.name}</div>
                    <div class="wizard-backend-detail">${detail}</div>
                </div>
            </div>`;
        }).join('');
    } catch {
        container.innerHTML = '<div class="wizard-loading">Could not detect backends.</div>';
    }
}

// === Shared exclusion directory picker ===

async function pickAndAddExclusion(chipContainerId) {
    const container = document.getElementById(chipContainerId);
    if (!container) return;

    // Open native folder picker
    let dirPath;
    try {
        const res = await fetch('/api/pick-directory');
        const data = await res.json();
        dirPath = data.path;
    } catch { return; }
    if (!dirPath) return; // user cancelled

    // Check for duplicates
    if (container.querySelector(`[data-path="${CSS.escape(dirPath)}"]`)) return;

    // Get message count
    let messageCount = 0;
    try {
        const res = await fetch('/api/exclusion-count', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: dirPath }),
        });
        const data = await res.json();
        messageCount = data.messageCount || 0;
    } catch { /* count unavailable */ }

    addExclusionChip(container, dirPath, messageCount);
}

function addExclusionChip(container, dirPath, messageCount) {
    const chip = document.createElement('div');
    chip.className = 'exclusion-chip';
    chip.dataset.path = dirPath;
    // Show just the last directory name for brevity
    const shortName = dirPath.split('/').pop() || dirPath;
    const countLabel = messageCount > 0 ? `${messageCount.toLocaleString()} messages` : 'no data found';
    chip.innerHTML = `
        <span class="exclusion-chip-path" title="${dirPath}">${shortName}</span>
        <span class="exclusion-chip-count">${countLabel}</span>
        <button class="exclusion-chip-remove" type="button" aria-label="Remove">&times;</button>
    `;
    chip.querySelector('.exclusion-chip-remove').addEventListener('click', () => chip.remove());
    container.appendChild(chip);
}

function getExclusionPaths(chipContainerId) {
    const container = document.getElementById(chipContainerId);
    if (!container) return [];
    return [...container.querySelectorAll('.exclusion-chip')].map(c => c.dataset.path);
}

async function loadExclusionChips(chipContainerId, exclusions) {
    const container = document.getElementById(chipContainerId);
    if (!container) return;
    container.innerHTML = '';
    for (const dirPath of exclusions) {
        // Get count for each existing exclusion
        let messageCount = 0;
        try {
            const res = await fetch('/api/exclusion-count', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: dirPath }),
            });
            const data = await res.json();
            messageCount = data.messageCount || 0;
        } catch { /* skip */ }
        addExclusionChip(container, dirPath, messageCount);
    }
}

// === Wizard step 2 ===

function wizardBuildConfig() {
    const container = document.getElementById('wizardConfig');
    const exclusionsDiv = document.getElementById('wizardExclusions');
    if (!container) return;

    const available = wizardBackendData.filter(b => b.status === 'available' && b.supported !== false);
    fetch('/api/config')
        .then((res) => res.json())
        .then((config) => {
            container.innerHTML = available.map(b => {
                const savedToggle = config.backends?.[b.id];
                const checked = savedToggle ? savedToggle.enabled !== false : true;
                return `
                    <label class="wizard-toggle">
                        <input type="checkbox" data-backend="${b.id}" ${checked ? 'checked' : ''}>
                        ${b.name}
                    </label>
                `;
            }).join('');

            if (available.some(b => b.id === 'claude_code') && exclusionsDiv) {
                exclusionsDiv.style.display = 'block';
                loadExclusionChips('wizardExclusionChips', config.backends?.claude_code?.exclusions ?? []);
            } else if (exclusionsDiv) {
                exclusionsDiv.style.display = 'none';
            }
        })
        .catch(() => {
            container.innerHTML = available.map(b => `
                <label class="wizard-toggle">
                    <input type="checkbox" data-backend="${b.id}" checked>
                    ${b.name}
                </label>
            `).join('');
        });
}

async function wizardSaveConfig() {
    const toggles = {};
    document.querySelectorAll('#wizardConfig input[type=checkbox]').forEach(cb => {
        toggles[cb.dataset.backend] = { enabled: cb.checked, exclusions: [] };
    });
    if (toggles.claude_code) {
        toggles.claude_code.exclusions = getExclusionPaths('wizardExclusionChips');
    }
    try {
        await fetch('/api/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backends: toggles }),
        });
    } catch { /* best effort */ }
}

function wizardRunPipeline() {
    const doneBtn = document.getElementById('wizardDone');
    const tracker = createWizardPipelineTracker();
    let completed = false;
    tracker.begin();

    const evtSource = new EventSource('/api/pipeline/stream');

    evtSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        tracker.advance(data.stage, data.detail, data.progress);
    });

    evtSource.addEventListener('complete', (e) => {
        completed = true;
        evtSource.close();
        const data = JSON.parse(e.data);
        tracker.finish(data.stats.totalMessages);
        if (doneBtn) doneBtn.style.display = 'inline-block';

        // Stash metrics so dashboard can render immediately
        if (data.metrics) {
            sourceViews = data.metrics.source_views || { both: data.metrics };
            if (!sourceViews.claude_code && sourceViews.claude) sourceViews.claude_code = sourceViews.claude;
            sourceViews.both = sourceViews.both || data.metrics;
            defaultSource = data.metrics.default_view || 'both';
            if (defaultSource === 'claude' && sourceViews.claude_code) defaultSource = 'claude_code';
        }
    });

    evtSource.addEventListener('pipeline_error', (e) => {
        completed = true;
        evtSource.close();
        const data = JSON.parse(e.data);
        tracker.fail(data.message);
        if (doneBtn) {
            doneBtn.textContent = 'Continue Anyway';
            doneBtn.style.display = 'inline-block';
        }
    });

    // Built-in error fires on connection close — ignore if we already completed
    evtSource.onerror = () => {
        if (!completed) {
            evtSource.close();
            tracker.fail('Connection lost — check terminal for details.');
            if (doneBtn) {
                doneBtn.textContent = 'Continue Anyway';
                doneBtn.style.display = 'inline-block';
            }
        }
    };
}

function switchWizardStep(step) {
    document.querySelectorAll('.wizard-step').forEach(s => {
        const n = Number(s.dataset.step);
        s.classList.toggle('active', n === step);
        s.classList.toggle('done', n < step);
    });
    document.querySelectorAll('.wizard-page').forEach(p => p.classList.remove('active'));
    document.getElementById(`wizardStep${step}`)?.classList.add('active');
}

// Wire wizard buttons
document.getElementById('wizardAddExclusion')?.addEventListener('click', () => pickAndAddExclusion('wizardExclusionChips'));
document.getElementById('wizardNext1')?.addEventListener('click', () => {
    wizardBuildConfig();
    switchWizardStep(2);
});
document.getElementById('wizardBack2')?.addEventListener('click', () => switchWizardStep(1));
document.getElementById('wizardNext2')?.addEventListener('click', async () => {
    const checked = document.querySelectorAll('#wizardConfig input[type=checkbox]:checked');
    if (checked.length === 0) {
        alert('Please enable at least one backend to analyze.');
        return;
    }
    await wizardSaveConfig();
    switchWizardStep(3);
    wizardRunPipeline();
});
document.getElementById('wizardDone')?.addEventListener('click', async () => {
    // Mark setup complete
    try {
        await fetch('/api/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hasCompletedSetup: true }),
        });
    } catch { /* best effort */ }

    // Hide wizard
    const wizard = document.getElementById('setupWizard');
    if (wizard) { wizard.classList.remove('active'); document.body.style.overflow = ''; }

    // Render dashboard with stashed metrics
    applyBranding({});
    initSourceFilter();
});

// === Init ===

async function init() {
    initThemeToggle();
    fixLocalLinks();
    initTabs();
    initCardTilt();

    // Re-render trend chart on theme toggle
    document.getElementById('themeToggle')?.addEventListener('click', () => {
        requestAnimationFrame(() => {
            if (activeView) initTrendChart(activeView);
        });
    });

    // Show mobile nav on small screens
    const mobileNav = document.getElementById('mobileNav');
    if (mobileNav && window.innerWidth <= 640) {
        mobileNav.style.display = 'flex';
    }

    // Check if wizard is needed
    const wizardActive = await initWizard();
    if (wizardActive) return;

    fetchDetectedBackends({ useCache: true }).catch(() => {});

    try {
        const response = await fetch('./metrics.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const metrics = await response.json();

        sourceViews = metrics.source_views || { both: metrics };
        if (!sourceViews.claude_code && sourceViews.claude) sourceViews.claude_code = sourceViews.claude;
        sourceViews.both = sourceViews.both || metrics;

        defaultSource = metrics.default_view || 'both';
        if (defaultSource === 'claude' && sourceViews.claude_code) defaultSource = 'claude_code';

        applyBranding(metrics);
        initSourceFilter();
    } catch (err) {
        console.warn('Could not load metrics.json:', err.message);
    }
}

init();
