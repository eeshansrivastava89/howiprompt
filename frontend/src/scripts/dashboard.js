// dashboard.js — Dashboard interactivity: source filtering, heatmap, SVG trends, player card, share
// Loads metrics.json via fetch() at runtime — no build-time data injection.

import { initThemeToggle } from './theme.js';

let sourceViews = {};
let defaultSource = 'both';
let activeSourceKey = 'both';
let activeView = null;
let branding = {};

const sourceFilter = document.getElementById('sourceFilter');
const sourceFilterMobile = document.getElementById('sourceFilterMobile');
const sourceStorageKey = 'dashboard-source-filter';

// === Formatting helpers ===

function formatNumber(value) {
    return (Number(value) || 0).toLocaleString();
}

function formatCompactK(value) {
    return `${Math.round((Number(value) || 0) / 1000)}K`;
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
    architect: '/images/card_architect.png',
    explorer: '/images/card_explorer.png',
    commander: '/images/card_commander.png',
    partner: '/images/card_partner.png',
    delegator: '/images/card_delegator.png',
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
    const personaType = persona.type || 'architect';
    if (cardTop) {
        cardTop.style.backgroundImage = `url('${CARD_IMAGES[personaType] || CARD_IMAGES.architect}')`;
    }

    const personaName = document.getElementById('personaName');
    if (personaName) personaName.textContent = persona.name || 'No Persona';

    const personaDesc = document.getElementById('personaDescription');
    if (personaDesc) personaDesc.textContent = persona.description || 'Not enough data.';

    // Use latest weekly NLP scores for donuts (matches trend chart)
    const weeklyRollups = view?.trends?.weekly_rollups || [];
    const latestWeek = weeklyRollups.length > 0 ? weeklyRollups[weeklyRollups.length - 1] : null;
    const latestNlp = latestWeek?.nlp || {};
    const radar = persona.radar || {};

    setDonut('donutPrecision', 'valPrecision', latestNlp.precision ?? radar.precision);
    setDonut('donutTenacity', 'valTenacity', latestNlp.tenacity ?? radar.tenacity);
    setDonut('donutCuriosity', 'valCuriosity', latestNlp.curiosity ?? radar.curiosity);
    setDonut('donutTrust', 'valTrust', latestNlp.trust ?? radar.trust);

    const hitlScore = nlp?.hitl_score?.avg_score;
    const vibeScore = nlp?.vibe_coder_index?.avg_score;
    const politePct = politeness?.pct ?? 0;

    const el = (id) => document.getElementById(id);
    if (el('cardHitl')) el('cardHitl').textContent = hitlScore != null ? Math.round(hitlScore) : '--';
    if (el('cardVibe')) el('cardVibe').textContent = vibeScore != null ? Math.round(vibeScore) : '--';
    if (el('cardPolite')) el('cardPolite').textContent = `${Math.round(politePct)}%`;
    if (el('cardHitlBar')) el('cardHitlBar').style.width = `${Math.min(hitlScore ?? 0, 100)}%`;
    if (el('cardVibeBar')) el('cardVibeBar').style.width = `${Math.min(vibeScore ?? 0, 100)}%`;
    if (el('cardPoliteBar')) el('cardPoliteBar').style.width = `${Math.min(politePct, 100)}%`;

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

        (data[dayIndex] || []).forEach((value) => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            if (value > 0) {
                const intensity = Math.ceil((value / maxVal) * 5);
                cell.classList.add('l' + Math.min(intensity, 5));
            }
            heatmap.appendChild(cell);
        });
    });
}

// === SVG Trend Chart ===

let trendData = {};
let activeTrendMetric = 'hitl';

const TREND_METRIC_CONFIG = {
    hitl:      { key: 'hitl_score',          label: 'Human in the Loop', suffix: '', desc: 'How actively you steer AI output. Higher = more hands-on review and iteration.' },
    vibe:      { key: 'vibe_coder_index',    label: 'Vibe Coder Index',  suffix: '', desc: 'Engineer vs. vibe coder spectrum. Higher = more structured, spec-driven prompting.' },
    polite:    { key: 'politeness_per_100',   label: 'Politeness',        suffix: '', desc: 'How often you say please, thanks, and sorry per 100 prompts.' },
    precision: { key: 'precision',           label: 'Precision',         suffix: '', desc: 'How specific and detailed your prompts are. Higher = more context and constraints.' },
    curiosity: { key: 'curiosity',           label: 'Curiosity',         suffix: '', desc: 'How often you explore, ask questions, and investigate new directions.' },
    tenacity:  { key: 'tenacity',            label: 'Tenacity',          suffix: '', desc: 'How persistently you iterate and refine. Higher = longer sessions, more follow-ups.' },
    trust:     { key: 'trust',               label: 'Trust',             suffix: '', desc: 'How much autonomy you give AI. Higher = fewer corrections, more delegation.' },
};

function extractTrendPoints(weekly, metricKey) {
    return weekly.map((w) => {
        // NLP weekly scores (hitl_score, vibe_coder_index, precision, etc.)
        if (w.nlp && w.nlp[metricKey] != null) return Math.round(w.nlp[metricKey]);
        // Style metrics (politeness_per_100, backtrack_per_100, etc.)
        if (w.style && w.style[metricKey] != null) return Math.round(w.style[metricKey]);
        return null;
    });
}

function initSvgTrend(view) {
    const trends = view?.trends || {};
    const weekly = trends?.weekly_rollups || [];
    if (weekly.length < 2) return;

    const ns = 'http://www.w3.org/2000/svg';
    const svgEl = document.getElementById('trendSvg');
    const areaEl = document.getElementById('trendArea');
    const lineEl = document.getElementById('trendLine');
    const guideEl = document.getElementById('trendGuide');
    const dotEnd = document.getElementById('trendDotEnd');
    const dotHover = document.getElementById('trendDotHover');
    const dotGlow = document.getElementById('trendDotGlow');
    const hitGroup = document.getElementById('trendHitAreas');
    const tooltip = document.getElementById('trendTooltip');
    const valEl = document.getElementById('trendVal');
    const labelEl = document.getElementById('trendLabel');

    if (!svgEl || !areaEl || !lineEl) return;

    const vw = 600, vh = 140, pad = 20;

    // Build week labels
    const weekLabels = weekly.map((w, i) => {
        if (i === weekly.length - 1) return 'Now';
        const ago = weekly.length - 1 - i;
        return `${ago}w ago`;
    });

    // Build data for all metrics
    trendData = {};
    for (const [key, config] of Object.entries(TREND_METRIC_CONFIG)) {
        const points = extractTrendPoints(weekly, config.key);
        // Filter out nulls — only include if we have data
        const hasData = points.some((p) => p != null);
        if (hasData) {
            trendData[key] = {
                points: points.map((p) => p ?? 0),
                label: config.label,
                suffix: config.suffix,
            };
        }
    }

    // If first metric has no data, find one that does
    if (!trendData[activeTrendMetric]) {
        activeTrendMetric = Object.keys(trendData)[0] || 'hitl';
    }

    function render() {
        const d = trendData[activeTrendMetric];
        if (!d) return;
        const pts = d.points;
        const min = Math.min(...pts) - 5;
        const max = Math.max(...pts) + 5;
        const range = max - min || 1;
        const stepX = (vw - pad * 2) / (pts.length - 1);

        const coords = pts.map((v, i) => ({
            x: pad + i * stepX,
            y: vh - pad - ((v - min) / range) * (vh - pad * 2),
        }));

        areaEl.setAttribute('points',
            `${coords[0].x},${vh} ` + coords.map((c) => `${c.x},${c.y}`).join(' ') + ` ${coords[coords.length - 1].x},${vh}`
        );
        lineEl.setAttribute('points', coords.map((c) => `${c.x},${c.y}`).join(' '));

        const last = coords[coords.length - 1];
        dotEnd.setAttribute('cx', last.x);
        dotEnd.setAttribute('cy', last.y);

        dotHover.style.display = 'none';
        dotGlow.style.display = 'none';
        guideEl.style.display = 'none';
        tooltip.style.opacity = '0';

        // Update SVG colors based on current accent
        const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#8839ef';
        lineEl.setAttribute('stroke', accent);
        dotEnd.setAttribute('fill', accent);
        dotHover.setAttribute('fill', accent);
        dotGlow.setAttribute('stroke', accent.replace(')', ',0.3)').replace('rgb', 'rgba'));

        hitGroup.innerHTML = '';
        coords.forEach((c, i) => {
            const rect = document.createElementNS(ns, 'rect');
            rect.setAttribute('x', c.x - stepX / 2);
            rect.setAttribute('y', 0);
            rect.setAttribute('width', stepX);
            rect.setAttribute('height', vh);
            rect.setAttribute('fill', 'transparent');
            rect.style.cursor = 'crosshair';

            rect.addEventListener('mouseenter', () => {
                dotHover.setAttribute('cx', c.x);
                dotHover.setAttribute('cy', c.y);
                dotGlow.setAttribute('cx', c.x);
                dotGlow.setAttribute('cy', c.y);
                dotHover.style.display = '';
                dotGlow.style.display = '';
                guideEl.setAttribute('x1', c.x);
                guideEl.setAttribute('y1', c.y + 12);
                guideEl.setAttribute('x2', c.x);
                guideEl.setAttribute('y2', vh);
                guideEl.style.display = '';
                const svgRect = svgEl.getBoundingClientRect();
                const pctX = c.x / vw;
                tooltip.style.left = (pctX * svgRect.width) + 'px';
                tooltip.style.opacity = '1';
                tooltip.textContent = `${weekLabels[i]}: ${pts[i]}${d.suffix}`;
            });

            rect.addEventListener('mouseleave', () => {
                dotHover.style.display = 'none';
                dotGlow.style.display = 'none';
                guideEl.style.display = 'none';
                tooltip.style.opacity = '0';
            });

            hitGroup.appendChild(rect);
        });
    }

    function setMetric(key) {
        if (!trendData[key]) return;
        activeTrendMetric = key;
        const d = trendData[key];
        const config = TREND_METRIC_CONFIG[key];
        const last = d.points[d.points.length - 1];
        valEl.textContent = last + d.suffix;
        labelEl.textContent = `${d.label} · ${weekly.length}-week trend`;
        const descEl = document.getElementById('trendDesc');
        if (descEl) descEl.textContent = config?.desc || '';
        document.querySelectorAll('.metric-tab').forEach((t) =>
            t.classList.toggle('active', t.dataset.metric === key)
        );
        render();
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

async function captureAndShare() {
    if (typeof html2canvas === 'undefined') return;
    const element = document.getElementById('playerCard');
    if (!element) return;
    const toast = document.getElementById('shareToast');
    try {
        const canvas = await html2canvas(element, {
            backgroundColor: null,
            scale: 2,
            useCORS: true,
        });
        canvas.toBlob(async (blob) => {
            if (!blob) return;
            const file = new File([blob], 'howiprompt-card.png', { type: 'image/png' });
            if (navigator.canShare && navigator.canShare({ files: [file] })) {
                try {
                    await navigator.share({ files: [file], title: 'How I Prompt Card' });
                    return;
                } catch (_) { /* user cancelled, fall through to download */ }
            }
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'howiprompt-card.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            if (toast) {
                toast.textContent = 'Card downloaded!';
                toast.classList.add('show');
                setTimeout(() => toast.classList.remove('show'), 2500);
            }
        }, 'image/png');
    } catch (err) {
        console.warn('Share capture failed:', err.message);
    }
}

document.getElementById('shareDashboard')?.addEventListener('click', captureAndShare);

// === Main renderView ===

function getView(sourceKey) {
    return sourceViews[sourceKey] || null;
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
    if (promptsSubtitle) promptsSubtitle.textContent = `${volume.avg_words_per_prompt ?? 0} words avg`;

    const conversationsValue = el('conversationsValue');
    if (conversationsValue) conversationsValue.textContent = formatNumber(volume.total_conversations);
    const conversationsSubtitle = el('conversationsSubtitle');
    if (conversationsSubtitle) conversationsSubtitle.textContent = `${convo.avg_turns ?? 0} turns avg`;

    const wordsTypedValue = el('wordsTypedValue');
    if (wordsTypedValue) wordsTypedValue.textContent = formatCompactK(volume.total_words_human);
    const wordsTypedSubtitle = el('wordsTypedSubtitle');
    if (wordsTypedSubtitle) wordsTypedSubtitle.textContent = `${formatCompactK(volume.total_words_assistant)} from assistants`;

    const nightOwlValue = el('nightOwlValue');
    if (nightOwlValue) nightOwlValue.textContent = `${temporal.night_owl_pct ?? 0}%`;

    // Heatmap meta
    const peakHourValue = el('peakHourValue');
    if (peakHourValue) peakHourValue.textContent = formatHour12(temporal.peak_hour);
    const peakDayValue = el('peakDayValue');
    if (peakDayValue) peakDayValue.textContent = temporal.peak_day || 'N/A';
    const responseRatioValue = el('responseRatioValue');
    if (responseRatioValue) responseRatioValue.textContent = `${view.response_ratio || 0}x`;

    // Player card
    renderPlayerCard(persona, view.nlp || {}, politeness, view);

    // Heatmap
    renderHeatmap(temporal.heatmap);

    // SVG trend chart
    initSvgTrend(view);
}

// === Source filter ===

function syncSourceSelectors(value) {
    if (sourceFilter) sourceFilter.value = value;
    if (sourceFilterMobile) sourceFilterMobile.value = value;
}

function initSourceFilter() {
    if (!sourceFilter) return;

    const available = ['both', 'claude_code', 'codex'].filter((key) => Boolean(getView(key)));
    const labelMap = { both: 'All', claude_code: 'Claude Code', codex: 'Codex' };
    [sourceFilter, sourceFilterMobile].filter(Boolean).forEach((select) => {
        select.innerHTML = '';
        for (const key of available) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = labelMap[key] || key;
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

function openMethodology() {
    const modal = document.getElementById('methodologyModal');
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeMethodology() {
    const modal = document.getElementById('methodologyModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

window.openMethodology = openMethodology;
window.closeMethodology = closeMethodology;

document.getElementById('methodologyModal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeMethodology();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMethodology();
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

    btn.classList.add('refreshing');
    const label = btn.querySelector('.refresh-label');
    if (label) label.textContent = 'Syncing...';

    try {
        const response = await fetch('/api/refresh', { method: 'POST' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const { metrics, stats } = await response.json();

        sourceViews = metrics.source_views || { both: metrics };
        if (!sourceViews.claude_code && sourceViews.claude) sourceViews.claude_code = sourceViews.claude;
        sourceViews.both = sourceViews.both || metrics;

        renderView(activeSourceKey);

        const toast = document.getElementById('shareToast');
        if (toast) {
            const msg = stats.newMessages > 0
                ? `Synced ${stats.newMessages} new message${stats.newMessages === 1 ? '' : 's'}`
                : 'Already up to date';
            toast.textContent = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2500);
        }
    } catch (err) {
        console.warn('Refresh failed:', err.message);
        const toast = document.getElementById('shareToast');
        if (toast) {
            toast.textContent = 'Refresh failed — is the server running?';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }
    } finally {
        btn.classList.remove('refreshing');
        const label = btn.querySelector('.refresh-label');
        if (label) label.textContent = 'Refresh';
    }
}

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

// === Leaderboard ===

const LEADERBOARD_API = 'https://leaderboard.howiprompt.com';
let leaderboardData = [];
let leaderboardLoaded = false;

async function fetchLeaderboard() {
    if (leaderboardLoaded) return;
    try {
        const response = await fetch(`${LEADERBOARD_API}/api/leaderboard`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        leaderboardData = await response.json();
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
    const sorted = [...leaderboardData].sort((a, b) => (b[sortKey] ?? 0) - (a[sortKey] ?? 0));

    body.innerHTML = sorted.map((entry, i) => `
        <tr${entry.is_you ? ' class="is-you"' : ''}>
            <td class="rank-col">${i + 1}</td>
            <td><strong>${escapeHtml(entry.display_name || 'Anonymous')}</strong></td>
            <td>${entry.hitl_score ?? '--'}</td>
            <td>${entry.vibe_index ?? '--'}</td>
            <td>${(entry.total_prompts ?? 0).toLocaleString()}</td>
            <td>${escapeHtml(entry.persona || '--')}</td>
        </tr>
    `).join('');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

document.getElementById('leaderboardSort')?.addEventListener('change', renderLeaderboard);

// === Submit to leaderboard ===

function getSubmissionPayload() {
    if (!activeView) return null;
    const v = activeView.volume || {};
    const nlp = activeView.nlp || {};
    const persona = activeView.persona || {};
    const norm = activeView.normalized || {};

    return {
        total_conversations: v.total_conversations || 0,
        total_prompts: v.total_human || 0,
        avg_words_per_prompt: v.avg_words_per_prompt || 0,
        politeness: Math.round(norm.politeness ?? 0),
        backtrack: Math.round(norm.backtrack ?? 0),
        question_rate: Math.round(norm.question_rate ?? 0),
        command_rate: Math.round(norm.command_rate ?? 0),
        hitl_score: Math.round(nlp.hitl_score?.avg_score ?? 0),
        vibe_index: Math.round(nlp.vibe_coder_index?.avg_score ?? 0),
        persona: persona.type || 'unknown',
        complexity_avg: Math.round(norm.complexity ?? 0),
        platform: activeSourceKey,
        tool_version: '2.0.0',
    };
}

function showSubmitModal() {
    const modal = document.getElementById('submitModal');
    const preview = document.getElementById('submitPreview');
    if (!modal || !preview) return;

    const payload = getSubmissionPayload();
    if (!payload) return;

    preview.innerHTML = Object.entries(payload)
        .map(([k, v]) => `<div style="display:flex;justify-content:space-between;padding:2px 0;"><span style="color:var(--muted)">${k}</span><span>${v}</span></div>`)
        .join('');

    modal.classList.add('active');
}

function hideSubmitModal() {
    document.getElementById('submitModal')?.classList.remove('active');
}

async function submitToLeaderboard() {
    const payload = getSubmissionPayload();
    const nameInput = document.getElementById('displayName');
    if (!payload || !nameInput) return;

    const displayName = nameInput.value.trim();
    if (!displayName) {
        nameInput.style.borderColor = 'red';
        nameInput.focus();
        return;
    }

    payload.display_name = displayName;
    hideSubmitModal();

    const toast = document.getElementById('leaderboardToast');
    try {
        const response = await fetch(`${LEADERBOARD_API}/api/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        if (toast) {
            toast.textContent = 'Scores submitted!';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2500);
        }

        leaderboardLoaded = false;
        fetchLeaderboard();
    } catch {
        if (toast) {
            toast.textContent = 'Submit failed — leaderboard service may not be deployed yet.';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }
    }
}

document.getElementById('submitLeaderboard')?.addEventListener('click', showSubmitModal);
document.getElementById('cancelSubmit')?.addEventListener('click', hideSubmitModal);
document.getElementById('confirmSubmit')?.addEventListener('click', submitToLeaderboard);
document.getElementById('submitModal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) hideSubmitModal();
});

// === Init ===

async function init() {
    initThemeToggle('dashboard-theme');
    fixLocalLinks();
    initTabs();
    initCardTilt();

    // Re-render SVG trend on theme toggle
    document.getElementById('themeToggle')?.addEventListener('click', () => {
        requestAnimationFrame(() => {
            if (activeView) initSvgTrend(activeView);
        });
    });

    // Show mobile nav on small screens
    const mobileNav = document.getElementById('mobileNav');
    if (mobileNav && window.innerWidth <= 640) {
        mobileNav.style.display = 'flex';
    }

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
