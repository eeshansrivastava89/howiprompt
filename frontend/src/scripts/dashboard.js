// dashboard.js — Dashboard interactivity: source filtering, heatmap, ApexCharts trends, player card, share
// Loads metrics.json via fetch() at runtime — no build-time data injection.

import { initThemeToggle } from './theme.js';
import { SOURCE_LABELS, SOURCE_ACCENTS, formatSourceLabel, getSourceDisplayName, formatHour12, formatDateRange, createDropdown } from './shared.js';
let sourceViews = {};
let defaultSource = 'both';
let activeSourceKey = 'both';
let activeView = null;
let branding = {};

const sourceBar = document.getElementById('sourceBar');
const sourceStorageKey = 'dashboard-source-filter';

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

    const vibeRaw = nlp?.vibe_coder_index?.avg_score;
    const vibeScore = vibeRaw != null ? 100 - vibeRaw : null;
    const politeScore = nlp?.politeness?.avg_score;

    const el = (id) => document.getElementById(id);
    if (el('cardVibe')) el('cardVibe').textContent = vibeScore != null ? Math.round(vibeScore) : '--';
    if (el('cardPolite')) el('cardPolite').textContent = politeScore != null ? Math.round(politeScore) : '--';
    if (el('cardVibeBar')) el('cardVibeBar').style.width = `${Math.min(vibeScore ?? 0, 100)}%`;
    if (el('cardPoliteBar')) el('cardPoliteBar').style.width = `${Math.min(politeScore, 100)}%`;

    const serial = el('cardSerial');
    if (serial) {
        const hash = String(Math.abs((persona.name || '').split('').reduce((a, c) => ((a << 5) - a) + c.charCodeAt(0), 0)) % 10000).padStart(4, '0');
        serial.textContent = `#${hash}`;
    }

    // Store explanation text for tooltip
    const fmtWhy = (entries) => (entries || []).map(e => {
        const arrow = e.contribution > 0 ? '\u2191' : '\u2193';
        return `${arrow} ${e.label}: ${e.stat}`;
    }).join('\n') || 'Not enough data';
    window._whyText = { vibe: fmtWhy(view.vibe_explanation), politeness: fmtWhy(view.politeness_explanation) };

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
let activeTrendMetric = 'activity';

const TREND_METRIC_CONFIG = {
    vibe:      { key: 'vibe_coder_index',    label: 'Vibe Coder Index',  suffix: '/100', desc: '<strong>Higher</strong> = vibe coder. <strong>Lower</strong> = engineer.', invert: true },
    polite:    { key: 'politeness',          label: 'Politeness',        suffix: '/100', desc: '<strong>Higher</strong> = warmer. <strong>Lower</strong> = sharper.', axisTop: 'Warmer', axisBottom: 'Sharper' },
    activity:  { key: '_prompts',            label: 'Activity',          suffix: '/wk', desc: '<strong>Higher</strong> = heavier usage. <strong>Lower</strong> = quieter weeks.' },
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
    const defEl = document.getElementById('trendMetricDef');

    if (!chartEl) return;

    if (typeof ApexCharts === 'undefined') {
        chartEl.innerHTML = '<p style="text-align:center;color:var(--muted);padding:40px 0;font-size:13px;">Chart library failed to load. Check your connection or firewall settings.</p>';
        return;
    }

    // Build week date labels
    const weekDates = weekly.map((w, index) => {
        const d = new Date(w.week_start || w.date);
        const base = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        const prev = index > 0 ? new Date(weekly[index - 1].week_start || weekly[index - 1].date) : null;
        const showYear = index === 0 || !prev || prev.getFullYear() !== d.getFullYear();
        return showYear ? `${base} '${String(d.getFullYear()).slice(-2)}` : base;
    });

    // Lifetime averages for headline display
    const nlp = view.nlp || {};
    const avgPromptsPerWeek = weekly.length > 0
        ? Math.round(weekly.reduce((s, w) => s + w.prompts, 0) / weekly.length)
        : null;
    const lifetimeAvg = {
        vibe: nlp.vibe_coder_index?.avg_score,
        polite: nlp.politeness?.avg_score,
        activity: avgPromptsPerWeek,
    };

    // Build trendData structure
    trendData = {};
    for (const [key, config] of Object.entries(TREND_METRIC_CONFIG)) {
        const rawPoints = extractTrendPoints(weekly, config.key);
        const points = config.invert ? rawPoints.map(p => p != null ? 100 - p : null) : rawPoints;
        const hasData = points.some((p) => p != null);
        if (hasData) {
            const rawLifetime = lifetimeAvg[key];
            const entry = {
                points: points.map((p) => p ?? 0),
                label: config.label,
                desc: config.desc,
                suffix: config.suffix,
                lifetime: config.invert && rawLifetime != null ? 100 - rawLifetime : rawLifetime,
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
                        if (pw.nlp && pw.nlp[config.key] != null) {
                            const val = Math.round(pw.nlp[config.key]);
                            return config.invert ? 100 - val : val;
                        }
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
        activeTrendMetric = Object.keys(trendData)[0] || 'vibe';
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

    function isStacked(metricKey, series) {
        return metricKey === 'activity' && series.length > 1;
    }

    function getYAxisBounds(metricKey, series) {
        if (metricKey === 'vibe' || metricKey === 'polite') {
            return { min: 0, max: 100, tickAmount: 4 };
        }

        let maxValue;
        if (isStacked(metricKey, series)) {
            // For stacked: sum per week across all series
            const weekCount = series[0]?.data?.length || 0;
            maxValue = 0;
            for (let w = 0; w < weekCount; w++) {
                let weekSum = 0;
                for (const s of series) weekSum += (s.data?.[w] ?? 0);
                if (weekSum > maxValue) maxValue = weekSum;
            }
        } else {
            const values = series.flatMap((s) => s.data || []).filter((v) => Number.isFinite(v));
            maxValue = values.length > 0 ? Math.max(...values) : 0;
        }
        const paddedMax = maxValue <= 0 ? 10 : Math.ceil(maxValue * 1.15);
        return { min: 0, max: paddedMax, tickAmount: 4 };
    }

    function getTickAmount() {
        if (window.innerWidth <= 1100) return 4;
        if (window.innerWidth <= 1400) return 5;
        return 7;
    }

    const isDark = document.documentElement.classList.contains('dark');
    const mutedColor = cssVar('--muted') || '#888';
    const borderColor = cssVar('--border') || '#ddd';
    const initial = buildSeriesAndColors(activeTrendMetric);
    const initialSuffix = trendData[activeTrendMetric]?.suffix || '';
    const initialYBounds = getYAxisBounds(activeTrendMetric, initial.series);

    const initialStacked = isStacked(activeTrendMetric, initial.series);
    const options = {
        chart: {
            type: 'area',
            stacked: initialStacked,
            height: '100%',
            fontFamily: "'DM Sans', system-ui, sans-serif",
            toolbar: { show: false },
            zoom: { enabled: false },
            background: 'transparent',
            animations: {
                enabled: false,
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
            tickAmount: getTickAmount(),
            axisBorder: { show: false },
            axisTicks: { show: false },
        },
        yaxis: {
            show: true,
            min: initialYBounds.min,
            max: initialYBounds.max,
            tickAmount: initialYBounds.tickAmount,
            forceNiceScale: false,
            axisBorder: { show: false },
            axisTicks: { show: false },
            labels: {
                style: { colors: mutedColor, fontSize: '11px' },
                formatter: (val) => activeTrendMetric === 'activity' ? formatCompact(Math.round(val)) : `${Math.round(val)}`,
            },
        },
        grid: {
            show: true,
            borderColor,
            strokeDashArray: 3,
            xaxis: { lines: { show: false } },
            yaxis: { lines: { show: true } },
            padding: { left: 12, right: 12, top: -8, bottom: 0 },
        },
        legend: { show: false },
        tooltip: {
            shared: true,
            intersect: false,
            theme: isDark ? 'dark' : 'light',
            custom: function({ series, seriesIndex, dataPointIndex, w }) {
                const suffix = trendData[activeTrendMetric]?.suffix || '';
                const stk = isStacked(activeTrendMetric, w.config.series);
                const rows = [];
                let total = 0;
                for (let i = 0; i < series.length; i++) {
                    const val = series[i]?.[dataPointIndex];
                    if (val == null || val === 0) continue;
                    total += val;
                    const color = w.config.colors[i] || '#666';
                    const name = w.config.series[i]?.name || '';
                    rows.push(`<div style="display:flex;align-items:center;gap:6px;padding:2px 0"><span style="width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></span><span style="flex:1">${name}</span><span style="font-weight:700">${Math.round(val)}${suffix}</span></div>`);
                }
                if (rows.length === 0) return '';
                if (stk && rows.length > 1) {
                    rows.push(`<div style="border-top:1px solid rgba(128,128,128,0.3);margin-top:2px;padding-top:4px;display:flex;justify-content:space-between;font-weight:700"><span>Total</span><span>${Math.round(total)}${suffix}</span></div>`);
                }
                const cat = w.config.xaxis?.categories?.[dataPointIndex] || '';
                const bg = isDark ? '#2a2420' : '#fff';
                const fg = isDark ? '#e0d6cc' : '#3c3226';
                return `<div style="background:${bg};color:${fg};border:1px solid ${isDark?'#3d342c':'#d9d2c9'};border-radius:8px;padding:8px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;min-width:120px"><div style="font-weight:700;margin-bottom:4px;color:${isDark?'#8a7d6f':'#6b5e50'}">${cat}</div>${rows.join('')}</div>`;
            },
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
        // Activity: show most recent week total. Others: lifetime average.
        let headlineRaw;
        if (key === 'activity') {
            // For stacked (All view), sum all series for the last week
            const lastIdx = d.points.length - 1;
            if (d.platforms && Object.keys(d.platforms).length > 1) {
                headlineRaw = Object.values(d.platforms).reduce((sum, pts) => sum + (pts[lastIdx] ?? 0), 0);
            } else {
                headlineRaw = d.points[lastIdx] ?? 0;
            }
        } else {
            headlineRaw = d.lifetime != null ? Math.round(d.lifetime) : d.points[d.points.length - 1];
        }
        valEl.textContent = (key === 'activity' ? formatCompact(headlineRaw) : headlineRaw) + d.suffix;
        labelEl.textContent = '';
        if (defEl) defEl.innerHTML = d.desc || '';

        // Set headline color
        const trendPanel = valEl.closest('.trend-panel');
        if (trendPanel) {
            trendPanel.style.setProperty('--trend-color', resolveAccent());
        }

        const { series, colors, showLegend } = buildSeriesAndColors(key);
        const suffix = d.suffix || '';
        const yBounds = getYAxisBounds(key, series);

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

        const stacked = isStacked(key, series);
        trendChartInstance.updateOptions({
            series,
            colors,
            chart: {
                stacked,
                animations: { enabled: false },
            },
            xaxis: {
                categories: weekDates,
                tickAmount: getTickAmount(),
                labels: {
                    style: { colors: mutedColor, fontSize: '11px' },
                    rotate: 0,
                    hideOverlappingLabels: true,
                },
                axisBorder: { show: false },
                axisTicks: { show: false },
            },
            yaxis: {
                show: true,
                min: yBounds.min,
                max: yBounds.max,
                tickAmount: yBounds.tickAmount,
                forceNiceScale: false,
                axisBorder: { show: false },
                axisTicks: { show: false },
                labels: {
                    style: { colors: mutedColor, fontSize: '11px' },
                    formatter: (val) => key === 'activity' ? formatCompact(Math.round(val)) : `${Math.round(val)}`,
                },
            },
        }, false, false, false);
    }

    // Metric dropdown
    const metricTabsEl = document.getElementById('metricTabs');
    if (metricTabsEl) {
        const metricItems = Object.entries(TREND_METRIC_CONFIG)
            .filter(([k]) => trendData[k])
            .map(([k, cfg]) => ({ key: k, label: cfg.label }));
        createDropdown({
            container: metricTabsEl,
            items: metricItems,
            selected: activeTrendMetric,
            placeholder: 'Metric',
            onSelect(key) { setMetric(key); },
        });
    }

    setMetric(activeTrendMetric);
}

// === 3D Tilt (desktop hover + mobile gyroscope) ===

function initCardTilt() {
    const card = document.getElementById('playerCard');
    const dock = card?.closest('.card-dock');
    if (!card || !dock) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const MAX_TILT = 12;

    // Desktop: mouse hover tilt
    dock.addEventListener('mousemove', (e) => {
        const r = dock.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width;
        const y = (e.clientY - r.top) / r.height;
        card.style.transform = `rotateY(${(x - 0.5) * MAX_TILT}deg) rotateX(${(0.5 - y) * MAX_TILT}deg)`;
    });
    dock.addEventListener('mouseleave', () => { card.style.transform = ''; });

    // Mobile: gyroscope tilt
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (!isTouchDevice || !window.DeviceOrientationEvent) return;

    const GYRO_TILT = 8;
    let gyroActive = false;
    let smoothX = 0, smoothY = 0;
    const LERP = 0.12;

    function handleOrientation(e) {
        const beta = e.beta;
        const gamma = e.gamma;
        if (beta == null || gamma == null) return;

        // Center around typical holding angle (~45° beta when upright)
        const rawY = Math.max(-25, Math.min(25, beta - 45)) / 25;
        const rawX = Math.max(-25, Math.min(25, gamma)) / 25;

        smoothX += (rawX - smoothX) * LERP;
        smoothY += (rawY - smoothY) * LERP;

        card.style.transform = `rotateY(${smoothX * GYRO_TILT}deg) rotateX(${-smoothY * GYRO_TILT}deg)`;
    }

    function startGyro() {
        if (gyroActive) return;
        gyroActive = true;
        window.addEventListener('deviceorientation', handleOrientation);
    }

    // iOS 13+ requires explicit permission from a user gesture
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        // Try on any tap on the card or the page
        function requestGyroPermission() {
            DeviceOrientationEvent.requestPermission()
                .then((state) => {
                    if (state === 'granted') startGyro();
                })
                .catch(() => {});
            // Remove both listeners once triggered
            card.removeEventListener('click', requestGyroPermission);
            document.removeEventListener('touchend', requestGyroPermission);
        }
        // Attach to card click (most reliable user gesture) and document touchend as fallback
        card.addEventListener('click', requestGyroPermission);
        document.addEventListener('touchend', requestGyroPermission, { once: true });
    } else {
        // Android, older iOS — just start
        startGyro();
    }
}

// === Main renderView ===

function getView(sourceKey) {
    return sourceViews[sourceKey] || null;
}

function getAvailableSourceKeys() {
    return Object.keys(sourceViews)
        .filter((key) => key === 'both' || Boolean(getView(key)))
        .sort((a, b) => {
            if (a === 'both') return 1;
            if (b === 'both') return -1;
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

let sourceDropdown = null;

function initSourceFilter() {
    if (!sourceBar) return;

    const available = getAvailableSourceKeys();
    const items = available.map(key => ({
        key,
        label: formatSourceLabel(key),
        color: SOURCE_ACCENTS[key],
    }));

    let selected = localStorage.getItem(sourceStorageKey) || defaultSource;
    if (selected === 'claude' && available.includes('claude_code')) selected = 'claude_code';
    if (!available.includes(selected)) {
        selected = available.includes('both') ? 'both' : available[0] || 'both';
    }

    sourceDropdown = createDropdown({
        container: sourceBar,
        items,
        selected,
        placeholder: 'Source',
        onSelect(key) {
            localStorage.setItem(sourceStorageKey, key);
            renderView(key);
        },
    });

    sourceDropdown.select(selected);
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
        const isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        const emptyMsg = isRefreshing
            ? 'Checking installed backends…'
            : isLocal ? 'No backend settings yet.' : 'Settings are available when running locally via <code style="font-size:12px">npx @eeshans/howiprompt</code>';
        container.innerHTML = `<div class="wizard-loading">${emptyMsg}</div>`;
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

// Why tooltip
(function() {
    const tip = document.getElementById('whyTip');
    if (!tip) return;
    const show = (pill, key) => {
        const text = window._whyText?.[key];
        if (!text) return;
        tip.textContent = text;
        tip.style.opacity = '0';
        tip.style.left = '0';
        tip.style.top = '0';
        // Force layout so offsetWidth is correct
        void tip.offsetWidth;
        const rect = pill.getBoundingClientRect();
        const tipW = tip.offsetWidth;
        const tipH = tip.offsetHeight;
        let left = rect.left + rect.width / 2 - tipW / 2;
        // Clamp to viewport
        left = Math.max(8, Math.min(left, window.innerWidth - tipW - 8));
        const top = rect.top - tipH - 10;
        tip.style.left = left + 'px';
        tip.style.top = top + 'px';
        tip.style.opacity = '1';
    };
    const hide = () => { tip.style.opacity = '0'; };
    ['whyVibe', 'whyPolite'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const key = id === 'whyVibe' ? 'vibe' : 'politeness';
        el.addEventListener('mouseenter', () => show(el, key));
        el.addEventListener('mouseleave', hide);
    });
})();

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

let refreshRunning = false;

async function handleRefresh() {
    if (refreshRunning) return;
    refreshRunning = true;

    const modal = document.getElementById('refreshModal');
    const log = document.getElementById('refreshLog');
    const bar = document.getElementById('refreshProgressBar');
    const resultEl = document.getElementById('refreshModalResult');
    const closeBtn = document.getElementById('refreshModalClose');

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
        refreshRunning = false;
    };

    // Reuse same SSE log pattern as wizard
    const stages = ['sync', 'parse', 'insert', 'nlp', 'style', 'scoring', 'metrics'];
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
        if (stats.scored > 0) {
            lines.push(`<span class="result-num">${stats.scored}</span> prompts scored`);
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

// === Setup Wizard ===

let wizardBackendData = [];
const PIPELINE_STAGE_ORDER = [
    { key: 'boot', label: 'Prepare' },
    { key: 'sync', label: 'Sync Sources' },
    { key: 'parse', label: 'Parse Chats' },
    { key: 'insert', label: 'Store Messages' },
    { key: 'nlp', label: 'Language Scores' },
    { key: 'style', label: 'Persona Scoring' },
    { key: 'scoring', label: 'Behavior Scores' },
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
            const ceiling = stageKey === 'scoring' ? 95 : 88;
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
    initCardTilt();

    // Re-render trend chart on theme toggle
    document.getElementById('themeToggle')?.addEventListener('click', () => {
        requestAnimationFrame(() => {
            if (activeView) initTrendChart(activeView);
        });
    });

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
