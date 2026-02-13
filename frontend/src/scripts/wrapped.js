// wrapped.js — Wrapped experience interactivity
// Heatmap rendering, counter animations, intersection observer, theme toggle
// Loads metrics.json via fetch() at runtime.

import { initThemeToggle } from './theme.js';

let metricsData = null;

// === Heatmap ===

function renderHeatmap(heatmapData) {
    const data = Array.isArray(heatmapData) ? heatmapData : [];
    const heatmapContainer = document.getElementById('heatmap');
    if (!heatmapContainer || data.length === 0) return;

    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const maxVal = Math.max(1, ...data.flat());

    function getHeatColor(value) {
        if (value === 0) return 'bg-bg';
        const intensity = value / maxVal;
        if (intensity < 0.2) return 'bg-accent/20';
        if (intensity < 0.4) return 'bg-accent/40';
        if (intensity < 0.6) return 'bg-accent/60';
        if (intensity < 0.8) return 'bg-accent/80';
        return 'bg-accent';
    }

    heatmapContainer.innerHTML = '';
    data.forEach((row, dayIndex) => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'flex items-center gap-[2px]';
        const dayLabel = document.createElement('span');
        dayLabel.className = 'w-12 text-xs text-muted shrink-0';
        dayLabel.textContent = days[dayIndex];
        rowDiv.appendChild(dayLabel);
        const cellsContainer = document.createElement('div');
        cellsContainer.className = 'flex-1 grid grid-cols-24 gap-[2px]';
        row.forEach((value, hourIndex) => {
            const cell = document.createElement('div');
            cell.className = `heatmap-cell aspect-square rounded-sm ${getHeatColor(value)} cursor-pointer`;
            cell.title = `${days[dayIndex]} ${hourIndex}:00 - ${value} prompts`;
            cellsContainer.appendChild(cell);
        });
        rowDiv.appendChild(cellsContainer);
        heatmapContainer.appendChild(rowDiv);
    });
}

// === Counter animation ===

function animateCounter(element, duration = 1500) {
    const target = parseInt(element.dataset.target);
    if (isNaN(target)) return;
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(target * easeOut);
        element.textContent = current.toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
        else element.textContent = target.toLocaleString();
    }
    requestAnimationFrame(update);
}

// === Intersection observer ===

function initAnimations() {
    const animatedElements = new Set();
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !animatedElements.has(entry.target)) {
                animatedElements.add(entry.target);
                entry.target.classList.add('animate-fade-in-up');
                entry.target.querySelectorAll('.counter').forEach(counter => {
                    if (counter.dataset.target) animateCounter(counter);
                });
            }
        });
    }, { threshold: 0.3 });
    document.querySelectorAll('section > div').forEach(el => observer.observe(el));
}

// === Fix dashboard link for local vs production ===

function fixLocalLinks() {
    const isLocal = location.protocol === 'file:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (isLocal) {
        document.querySelectorAll('.dashboard-link').forEach(link => {
            const currentPath = location.pathname;
            // Handle both /wrapped/ (trailing slash) and /wrapped (no trailing slash)
            if (currentPath.includes('/wrapped')) {
                const wrappedIdx = currentPath.indexOf('/wrapped');
                link.href = currentPath.substring(0, wrappedIdx) + '/index.html';
            } else {
                link.href = '/index.html';
            }
        });
    }
}

// === Formatting helpers ===

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

// === Hydrate all wrapped sections ===

function hydrateWrapped(m) {
    const v = m.volume || {};
    const t = m.temporal || {};
    const pol = m.politeness || {};
    const back = m.backtrack || {};
    const q = m.question || {};
    const cmd = m.command || {};
    const yr = m.youre_right || {};
    const p = m.persona || {};
    const cd = m.conversation_depth || {};
    const rr = m.response_ratio || 0;
    const dr = m.date_range || {};

    const dateRangeDisplay = formatDateRange(dr);
    const peakHour12h = formatHour12(t.peak_hour);

    // Politeness bar widths (relative to max)
    const polCounts = pol.counts || {};
    const polMax = Math.max(polCounts.please || 0, polCounts.sorry || 0, polCounts.thanks || 0, 1);
    const polPleasePct = Math.round((polCounts.please || 0) / polMax * 100);
    const polSorryPct = Math.round((polCounts.sorry || 0) / polMax * 100);
    const polThanksPct = Math.round((polCounts.thanks || 0) / polMax * 100);

    // Backtrack bar widths
    const backCounts = back.counts || {};
    const backMax = Math.max(backCounts.actually || 0, backCounts.wait || 0, 1);
    const backActuallyPct = Math.round((backCounts.actually || 0) / backMax * 100);
    const backWaitPct = Math.round((backCounts.wait || 0) / backMax * 100);

    const el = (id) => document.getElementById(id);
    const setText = (id, text) => { const e = el(id); if (e) e.textContent = text; };
    const setWidth = (id, pct) => { const e = el(id); if (e) e.style.width = `${pct}%`; };

    // Section 1: Cold Open
    setText('yrCount', yr.count || 0);
    setText('yrPerConvo', `${yr.per_conversation || 0}×`);
    setText('dateRangeDisplay', dateRangeDisplay);

    // Section 2: Numbers
    const counterEl = el('mainCounter');
    if (counterEl) {
        counterEl.dataset.target = v.total_human || 0;
        counterEl.textContent = (v.total_human || 0).toLocaleString();
    }
    setText('totalConversations', (v.total_conversations || 0).toLocaleString());
    setText('totalWordsK', `${Math.round((v.total_words_human || 0) / 1000)}K`);
    setText('avgTurns', cd.avg_turns || 0);
    setText('responseRatio', `${rr}x`);
    setText('maxTurns', cd.max_turns || 0);
    setText('deepDives', cd.deep_dives || 0);

    // Section 3: Temporal
    setText('peakHour', peakHour12h);
    setText('peakHourCount', `${t.peak_hour_count || 0} prompts`);
    setText('nightOwlPct', `${t.night_owl_pct || 0}%`);
    setText('peakDay', t.peak_day || 'N/A');
    setText('peakDayCount', `${t.peak_day_count || 0} prompts`);

    // Section 4: Prompt Style
    setText('polPer100', pol.per_100_prompts ?? 0);
    setText('polPleaseCount', polCounts.please || 0);
    setText('polSorryCount', polCounts.sorry || 0);
    setText('polThanksCount', polCounts.thanks || 0);
    setWidth('polPleaseBar', polPleasePct);
    setWidth('polSorryBar', polSorryPct);
    setWidth('polThanksBar', polThanksPct);

    setText('backPer100', back.per_100_prompts ?? 0);
    setText('backActuallyCount', backCounts.actually || 0);
    setText('backWaitCount', backCounts.wait || 0);
    setWidth('backActuallyBar', backActuallyPct);
    setWidth('backWaitBar', backWaitPct);

    setText('questionRate', `${q.rate || 0}%`);
    setText('questionCount', `${q.count || 0} questions`);
    setText('questionTotal', `of ${(v.total_human || 0).toLocaleString()}`);
    setWidth('questionBar', q.rate || 0);

    setText('commandRate', `${cmd.rate || 0}%`);
    setText('commandCount', `${cmd.count || 0} commands`);
    setWidth('commandBar', cmd.rate || 0);

    // Section 5: Conversation Patterns
    setText('quickAsks', cd.quick_asks || 0);
    setText('workingSessions', cd.working_sessions || 0);
    setText('deepDivesS5', cd.deep_dives || 0);
    setText('avgTurnsS5', cd.avg_turns || 0);
    setText('maxTurnsS5', `${cd.max_turns || 0} turns`);
    setText('responseRatioS5', `${rr}x`);
    setText('responseRatioLabel', rr);

    // Section 6: Persona
    setText('personaName', p.name || '--');
    setText('personaDesc', p.description || '');
    const traitsContainer = el('personaTraits');
    if (traitsContainer) {
        traitsContainer.innerHTML = '';
        (p.traits || []).forEach(trait => {
            const span = document.createElement('span');
            span.className = 'px-4 py-2 bg-border rounded-full text-sm font-medium';
            span.textContent = trait;
            traitsContainer.appendChild(span);
        });
    }
    setText('personaPolScore', p.scores?.politeness ?? 0);
    setText('personaBackScore', p.scores?.backtrack ?? 0);
    setText('personaQRate', `${p.scores?.question_rate ?? 0}%`);
    setText('personaCmdRate', `${p.scores?.command_rate ?? 0}%`);

    // Section 7: Terminal card
    setText('termDateRange', dateRangeDisplay.toUpperCase() + ' SUMMARY');
    setText('termPrompts', (v.total_human || 0).toLocaleString());
    setText('termConversations', v.total_conversations || 0);
    setText('termWordsK', `${Math.round((v.total_words_human || 0) / 1000)}K`);
    setText('termAvgTurns', cd.avg_turns || 0);
    setText('termMaxTurns', `${cd.max_turns || 0} turns`);
    setText('termDeepDives', cd.deep_dives || 0);
    setText('termPeakHour', peakHour12h);
    setText('termPeakDay', t.peak_day || 'N/A');
    setText('termNightOwl', `${t.night_owl_pct || 0}%`);
    setText('termYrCount', `${yr.count || 0}x`);
    setText('termPersonaName', `PERSONA: ${(p.name || '--').toUpperCase()}`);
    setText('termPersonaTraits', (p.traits || []).join(' • '));

    // Render heatmap
    renderHeatmap(t.heatmap);

    // Apply branding links
    const branding = m.branding || {};
    if (branding.site_url) {
        const brandingLink = el('brandingLink');
        if (brandingLink) brandingLink.href = branding.site_url;
    }
    if (branding.site_name) {
        setText('brandingSiteName', branding.site_name);
    }
    if (branding.github_repo) {
        document.querySelectorAll('.build-your-own-link').forEach(link => {
            link.href = branding.github_repo;
        });
    }
    if (branding.newsletter_url) {
        const nl = el('newsletterLink');
        if (nl) nl.href = branding.newsletter_url;
    }
}

// === Init ===

async function init() {
    initThemeToggle('theme');
    fixLocalLinks();
    initAnimations();

    try {
        const response = await fetch('./metrics.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        metricsData = await response.json();

        // For the wrapped page, use the claude_code view if available, else default
        const sourceViews = metricsData.source_views || {};
        const wrappedView = sourceViews.claude_code || sourceViews.both || metricsData;

        hydrateWrapped(wrappedView);
    } catch (err) {
        console.warn('Could not load metrics.json:', err.message);
    }
}

init();
