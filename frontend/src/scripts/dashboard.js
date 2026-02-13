// dashboard.js — Dashboard interactivity: source filtering, heatmap, ECharts trends, share, modal
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

// === Tag state helper ===

function setTagState(el, isHigh) {
    if (!el) return;
    el.classList.remove('high', 'low');
    el.classList.add(isHigh ? 'high' : 'low');
    el.textContent = isHigh ? 'High' : 'Low';
}

// === Persona traits ===

function renderTraits(traits) {
    const container = document.getElementById('personaTraits');
    if (!container) return;
    container.innerHTML = '';
    (traits || []).forEach((trait) => {
        const chip = document.createElement('span');
        chip.className = 'trait';
        chip.textContent = trait;
        container.appendChild(chip);
    });
}

// === Heatmap ===

function renderHeatmap(heatmapData) {
    const data = Array.isArray(heatmapData) ? heatmapData : [];
    const heatmap = document.getElementById('heatmap');
    if (!heatmap) return;
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    heatmap.innerHTML = `
        <div></div>
        <div class="heatmap-hours">
            <span>0</span><span></span><span></span><span>3</span><span></span><span></span>
            <span>6</span><span></span><span></span><span>9</span><span></span><span></span>
            <span>12</span><span></span><span></span><span>15</span><span></span><span></span>
            <span>18</span><span></span><span></span><span>21</span><span></span><span></span>
        </div>
    `;

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

// === Trend rendering (ECharts) ===

let chartInstances = {};

function formatDelta(deltaPct) {
    if (deltaPct === null || deltaPct === undefined || Number.isNaN(deltaPct)) {
        return { text: 'n/a', cls: 'flat' };
    }
    if (Math.abs(deltaPct) < 0.1) {
        return { text: '0.0%', cls: 'flat' };
    }
    if (deltaPct > 0) {
        return { text: `+${deltaPct}%`, cls: 'up' };
    }
    return { text: `${deltaPct}%`, cls: 'down' };
}

function isDarkMode() {
    return document.documentElement.classList.contains('dark');
}

function getChartColors() {
    const dark = isDarkMode();
    return {
        lineColor: '#f97316',
        areaStart: dark ? 'rgba(249, 115, 22, 0.3)' : 'rgba(249, 115, 22, 0.2)',
        areaEnd: 'rgba(249, 115, 22, 0)',
        textColor: dark ? '#86868b' : '#86868b',
        tooltipBg: dark ? '#1c1c1e' : '#ffffff',
        tooltipBorder: dark ? '#38383a' : '#d2d2d7',
        tooltipText: dark ? '#f5f5f7' : '#1d1d1f',
    };
}

function buildChartOption(dates, values, colors) {
    return {
        animation: true,
        animationDuration: 600,
        grid: { top: 8, right: 8, bottom: 24, left: 36 },
        xAxis: {
            type: 'category',
            data: dates,
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: {
                fontSize: 10,
                color: colors.textColor,
                formatter: (v) => {
                    const d = new Date(v);
                    return `${d.getMonth() + 1}/${d.getDate()}`;
                },
                interval: Math.max(0, Math.floor(dates.length / 5) - 1),
            },
        },
        yAxis: {
            type: 'value',
            splitLine: { show: false },
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { fontSize: 10, color: colors.textColor },
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: colors.tooltipBg,
            borderColor: colors.tooltipBorder,
            textStyle: { color: colors.tooltipText, fontSize: 12 },
            formatter: (params) => {
                const p = params[0];
                return `<strong>${p.axisValue}</strong><br/>${p.value.toFixed(1)}`;
            },
        },
        series: [{
            type: 'line',
            data: values,
            smooth: true,
            symbol: 'none',
            lineStyle: { color: colors.lineColor, width: 2.5 },
            areaStyle: {
                color: {
                    type: 'linear',
                    x: 0, y: 0, x2: 0, y2: 1,
                    colorStops: [
                        { offset: 0, color: colors.areaStart },
                        { offset: 1, color: colors.areaEnd },
                    ],
                },
            },
        }],
    };
}

function disposeTrendCharts() {
    Object.values(chartInstances).forEach((chart) => {
        if (chart && !chart.isDisposed()) chart.dispose();
    });
    chartInstances = {};
}

const TREND_CHART_CONFIG = [
    { id: 'chartQuestionRate', badge: 'badgeQuestionRate', key: 'question_rate_pct', deltaKey: 'question_rate_pct' },
    { id: 'chartCommandRate', badge: 'badgeCommandRate', key: 'command_rate_pct', deltaKey: 'command_rate_pct' },
    { id: 'chartPoliteness', badge: 'badgePoliteness', key: 'politeness_per_100', deltaKey: 'politeness_per_100' },
    { id: 'chartBacktrack', badge: 'badgeBacktrack', key: 'backtrack_per_100', deltaKey: 'backtrack_per_100' },
];

function initTrendCharts(view) {
    if (typeof echarts === 'undefined') return;
    disposeTrendCharts();

    const trends = view?.trends || {};
    const weekly = trends?.weekly_rollups || [];
    const deltas = trends?.deltas_7d_vs_30d || {};
    const colors = getChartColors();

    if (weekly.length < 2) return;

    const dates = weekly.map((r) => r.week_start);

    TREND_CHART_CONFIG.forEach(({ id, badge, key, deltaKey }) => {
        const container = document.getElementById(id);
        if (!container) return;

        const values = weekly.map((r) => Number(r.style?.[key] ?? 0));
        const chart = echarts.init(container);
        chart.setOption(buildChartOption(dates, values, colors));
        chartInstances[id] = chart;

        // Delta badge
        const badgeEl = document.getElementById(badge);
        if (badgeEl) {
            const d = formatDelta(deltas?.[deltaKey]?.delta_pct);
            badgeEl.textContent = d.text;
            badgeEl.className = `trend-badge ${d.cls}`;
        }
    });
}

function resizeTrendCharts() {
    Object.values(chartInstances).forEach((chart) => {
        if (chart && !chart.isDisposed()) chart.resize();
    });
}

window.addEventListener('resize', resizeTrendCharts);

function renderTrendBand(view) {
    initTrendCharts(view);
}

// === Share MVP ===

async function captureAndShare(element) {
    if (typeof html2canvas === 'undefined') return;
    const toast = document.getElementById('shareToast');
    try {
        const canvas = await html2canvas(element, {
            backgroundColor: isDarkMode() ? '#000000' : '#f5f5f7',
            scale: 2,
            useCORS: true,
        });
        canvas.toBlob(async (blob) => {
            if (!blob) return;
            const file = new File([blob], 'howiprompt-dashboard.png', { type: 'image/png' });
            if (navigator.canShare && navigator.canShare({ files: [file] })) {
                try {
                    await navigator.share({ files: [file], title: 'How I Prompt Dashboard' });
                    return;
                } catch (_) { /* user cancelled or not supported, fall through to download */ }
            }
            // Fallback: download
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'howiprompt-dashboard.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            if (toast) {
                toast.classList.add('show');
                setTimeout(() => toast.classList.remove('show'), 2500);
            }
        }, 'image/png');
    } catch (err) {
        console.warn('Share capture failed:', err.message);
    }
}

document.getElementById('shareDashboard')?.addEventListener('click', () => {
    const container = document.querySelector('.container');
    if (container) captureAndShare(container);
});

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
    const backtrack = view.backtrack || {};
    const question = view.question || {};
    const command = view.command || {};
    const persona = view.persona || {};
    const quadrant = persona.quadrant || {};
    const youreRight = view.youre_right || {};
    const responseRatio = view.response_ratio || 0;

    const el = (id) => document.getElementById(id);

    const dateRange = el('dateRange');
    if (dateRange) dateRange.textContent = formatDateRange(view.date_range);

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
    const peakHourValue = el('peakHourValue');
    if (peakHourValue) peakHourValue.textContent = formatHour12(temporal.peak_hour);
    const peakDayValue = el('peakDayValue');
    if (peakDayValue) peakDayValue.textContent = temporal.peak_day || 'N/A';
    const responseRatioValue = el('responseRatioValue');
    if (responseRatioValue) responseRatioValue.textContent = `${responseRatio}x`;

    const quick = convo.quick_asks || 0;
    const working = convo.working_sessions || 0;
    const deep = convo.deep_dives || 0;
    const totalConvos = quick + working + deep;
    const quickPct = totalConvos ? Math.round((quick / totalConvos) * 100) : 0;
    const workingPct = totalConvos ? Math.round((working / totalConvos) * 100) : 0;
    const deepPct = totalConvos ? Math.round((deep / totalConvos) * 100) : 0;

    const quickAsksValue = el('quickAsksValue');
    if (quickAsksValue) quickAsksValue.textContent = quick;
    const workingSessionsValue = el('workingSessionsValue');
    if (workingSessionsValue) workingSessionsValue.textContent = working;
    const deepDivesValue = el('deepDivesValue');
    if (deepDivesValue) deepDivesValue.textContent = deep;
    const quickAsksFill = el('quickAsksFill');
    if (quickAsksFill) quickAsksFill.style.width = `${quickPct}%`;
    const workingSessionsFill = el('workingSessionsFill');
    if (workingSessionsFill) workingSessionsFill.style.width = `${workingPct}%`;
    const deepDivesFill = el('deepDivesFill');
    if (deepDivesFill) deepDivesFill.style.width = `${deepPct}%`;
    const longestSessionValue = el('longestSessionValue');
    if (longestSessionValue) longestSessionValue.textContent = `${convo.max_turns || 0} turns`;

    const politenessCounts = politeness.counts || {};
    const backtrackCounts = backtrack.counts || {};
    const politenessValueEl = el('politenessValue');
    if (politenessValueEl) politenessValueEl.textContent = politeness.per_100_prompts ?? 0;
    const politenessDetail = el('politenessDetail');
    if (politenessDetail) politenessDetail.textContent = `please: ${politenessCounts.please || 0} · thanks: ${politenessCounts.thanks || 0}`;
    const backtrackValueEl = el('backtrackValue');
    if (backtrackValueEl) backtrackValueEl.textContent = backtrack.per_100_prompts ?? 0;
    const backtrackDetail = el('backtrackDetail');
    if (backtrackDetail) backtrackDetail.textContent = `actually: ${backtrackCounts.actually || 0} · wait: ${backtrackCounts.wait || 0}`;
    const questionsValue = el('questionsValue');
    if (questionsValue) questionsValue.textContent = `${question.rate ?? 0}%`;
    const questionsDetail = el('questionsDetail');
    if (questionsDetail) questionsDetail.textContent = `${question.count || 0} total`;
    const commandsValue = el('commandsValue');
    if (commandsValue) commandsValue.textContent = `${command.rate ?? 0}%`;
    const commandsDetail = el('commandsDetail');
    if (commandsDetail) commandsDetail.textContent = `${command.count || 0} total`;

    const personaName = el('personaName');
    if (personaName) personaName.textContent = persona.name || 'No Persona';
    const personaDescription = el('personaDescription');
    if (personaDescription) personaDescription.textContent = persona.description || 'Not enough data for persona classification.';
    renderTraits(persona.traits || []);
    const engagementScore = el('engagementScore');
    if (engagementScore) engagementScore.textContent = quadrant.engagement_score ?? 0;
    const politenessScore = el('politenessScore');
    if (politenessScore) politenessScore.textContent = quadrant.politeness_score ?? 0;
    setTagState(el('engagementTag'), Boolean(quadrant.high_engagement));
    setTagState(el('politenessTag'), Boolean(quadrant.high_politeness));

    const youreRightCount = el('youreRightCount');
    if (youreRightCount) youreRightCount.textContent = `×${youreRight.count || 0}`;
    const youreRightLabel = el('youreRightLabel');
    if (youreRightLabel) youreRightLabel.textContent = `${youreRight.per_conversation ?? 0}× per conversation`;

    renderHeatmap(temporal.heatmap);
    renderTrendBand(view);
}

// === Source filter ===

function syncSourceSelectors(value) {
    if (sourceFilter) sourceFilter.value = value;
    if (sourceFilterMobile) sourceFilterMobile.value = value;
}

function initSourceFilter() {
    if (!sourceFilter || !sourceFilterMobile) return;

    const available = ['both', 'claude_code', 'codex'].filter((key) => Boolean(getView(key)));
    [sourceFilter, sourceFilterMobile].forEach((select) => {
        Array.from(select.options).forEach((option) => {
            option.disabled = !available.includes(option.value);
        });
    });

    let selected = localStorage.getItem(sourceStorageKey) || defaultSource;
    if (selected === 'claude' && available.includes('claude_code')) {
        selected = 'claude_code';
    }
    if (!available.includes(selected)) {
        selected = available[0] || 'both';
    }

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
    sourceFilterMobile.addEventListener('change', handleChange);
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

// Expose to inline onclick handlers
window.openMethodology = openMethodology;
window.closeMethodology = closeMethodology;

document.getElementById('methodologyModal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeMethodology();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMethodology();
});

// === Fix wrapped link for local vs production ===

function fixLocalLinks() {
    const isLocal = location.protocol === 'file:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (isLocal) {
        document.querySelectorAll('a[href="/wrapped"]').forEach(link => {
            const currentPath = location.pathname;
            const basePath = currentPath.substring(0, currentPath.lastIndexOf('/'));
            const fileName = currentPath.substring(currentPath.lastIndexOf('/') + 1);
            if (fileName === 'dashboard.html') {
                link.href = basePath + '/index.html';
            } else {
                link.href = basePath + '/wrapped/index.html';
            }
        });
    }
}

// === Apply branding from metrics ===

function applyBranding(metricsData) {
    branding = metricsData.branding || {};
    const githubRepo = branding.github_repo || 'https://github.com/eeshansrivastava89/howiprompt';
    const newsletterUrl = branding.newsletter_url || 'https://0to1datascience.substack.com';

    // Update links if branding available
    const buildLink = document.getElementById('buildYourOwnLink');
    if (buildLink) buildLink.href = githubRepo;
    const buildMobileLink = document.getElementById('buildYourOwnMobileLink');
    if (buildMobileLink) buildMobileLink.href = githubRepo;
    const subscribeLink = document.getElementById('subscribeLink');
    if (subscribeLink) subscribeLink.href = newsletterUrl;
    const footerGithubLink = document.getElementById('footerGithubLink');
    if (footerGithubLink) footerGithubLink.href = githubRepo;
}

// === Init ===

async function init() {
    initThemeToggle('dashboard-theme');
    fixLocalLinks();

    // Re-render ECharts on theme toggle (dark/light switch)
    document.getElementById('themeToggle')?.addEventListener('click', () => {
        // Small delay so the class toggle has applied
        requestAnimationFrame(() => {
            if (activeView) initTrendCharts(activeView);
        });
    });

    try {
        const response = await fetch('./metrics.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const metrics = await response.json();

        sourceViews = metrics.source_views || { both: metrics };
        // Backward compat: "claude" → "claude_code"
        if (!sourceViews.claude_code && sourceViews.claude) {
            sourceViews.claude_code = sourceViews.claude;
        }
        sourceViews.both = sourceViews.both || metrics;

        defaultSource = metrics.default_view || 'both';
        if (defaultSource === 'claude' && sourceViews.claude_code) {
            defaultSource = 'claude_code';
        }

        applyBranding(metrics);
        initSourceFilter();
    } catch (err) {
        console.warn('Could not load metrics.json:', err.message);
        // Page remains with placeholder values
    }
}

init();
