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

    // Card shows lifetime averages (stable persona representation)
    const radar = persona.radar || {};

    setDonut('donutPrecision', 'valPrecision', radar.precision);
    setDonut('donutTenacity', 'valTenacity', radar.tenacity);
    setDonut('donutCuriosity', 'valCuriosity', radar.curiosity);
    setDonut('donutTrust', 'valTrust', radar.trust);

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
        heatmap.addEventListener('mouseover', (e) => {
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
        });
        heatmap.addEventListener('mouseleave', () => { tooltip.style.opacity = '0'; });
    }
}

// === SVG Trend Chart ===

let trendData = {};
let activeTrendMetric = 'hitl';

const TREND_METRIC_CONFIG = {
    hitl:      { key: 'hitl_score',          label: 'Human in the Loop', suffix: '', desc: 'How actively you steer AI output. Higher = more hands-on review and iteration.' },
    vibe:      { key: 'vibe_coder_index',    label: 'Vibe Coder Index',  suffix: '', desc: 'Engineer vs. vibe coder spectrum. Higher = more structured, spec-driven prompting.' },
    polite:    { key: 'politeness',          label: 'Politeness',        suffix: '', desc: 'How courteous and collaborative your tone is. Higher = warmer, more appreciative prompting style.' },
    activity:  { key: '_prompts',            label: 'Activity',          suffix: '', desc: 'Prompts per week.' },
};

function extractTrendPoints(weekly, metricKey) {
    return weekly.map((w) => {
        if (metricKey === '_prompts') return w.prompts ?? null;
        if (w.nlp && w.nlp[metricKey] != null) return Math.round(w.nlp[metricKey]);
        return null;
    });
}

function getPlatformColors() {
    const isDark = document.documentElement.classList.contains('dark');
    return {
        claude_code: { stroke: '#e67e22', label: 'Claude Code' },
        codex:       { stroke: isDark ? '#b0a596' : '#2c2c2c', label: 'Codex' },
    };
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16), g = parseInt(hex.slice(3,5), 16), b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function resolveAccent() {
    // If viewing a specific platform, use its color
    const platColor = getPlatformColors()[activeSourceKey]?.stroke;
    if (platColor) return platColor;
    // Otherwise use CSS accent
    return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#5c3d2e';
}

function initSvgTrend(view) {
    const trends = view?.trends || {};
    const weekly = trends?.weekly_rollups || [];
    if (weekly.length < 2) return;

    const weeklyByPlatform = trends?.weekly_by_platform || {};
    const isAllView = Object.keys(weeklyByPlatform).length > 0;

    const ns = 'http://www.w3.org/2000/svg';
    const svgEl = document.getElementById('trendSvg');
    const areaEl = document.getElementById('trendArea');
    const lineEl = document.getElementById('trendLine');
    const guideEl = document.getElementById('trendGuide');
    const dotEnd = document.getElementById('trendDotEnd');
    const dotHover = document.getElementById('trendDotHover');
    const dotGlow = document.getElementById('trendDotGlow');
    const hitGroup = document.getElementById('trendHitAreas');
    const platformGroup = document.getElementById('trendPlatformLines');
    const xAxisGroup = document.getElementById('trendXAxis');
    const tooltip = document.getElementById('trendTooltip');
    const valEl = document.getElementById('trendVal');
    const labelEl = document.getElementById('trendLabel');

    if (!svgEl || !areaEl || !lineEl) return;

    const vw = 600, vh = 140, pad = 20;

    // Build week date labels from week_start keys
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

    // Build weekly data for trend lines (aggregate)
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

            // Build per-platform series aligned to aggregate week keys
            if (isAllView) {
                const weekKeys = weekly.map((w) => w.week_start);
                entry.platforms = {};
                for (const [plat, platWeekly] of Object.entries(weeklyByPlatform)) {
                    const platByWeek = new Map(platWeekly.map((w) => [w.week_start, w]));
                    const platPoints = weekKeys.map((wk) => {
                        const pw = platByWeek.get(wk);
                        if (!pw) return null;
                        if (config.key === '_prompts') return pw.prompts ?? null;
                        if (pw.nlp && pw.nlp[config.key] != null) return Math.round(pw.nlp[config.key]);
                        return null;
                    });
                    if (platPoints.some((p) => p != null)) {
                        entry.platforms[plat] = platPoints;
                    }
                }
            }

            trendData[key] = entry;
        }
    }

    // If first metric has no data, find one that does
    if (!trendData[activeTrendMetric]) {
        activeTrendMetric = Object.keys(trendData)[0] || 'hitl';
    }

    // Render x-axis date labels
    function renderXAxis(stepX) {
        xAxisGroup.innerHTML = '';
        // Show every Nth label to avoid crowding
        const maxLabels = 8;
        const step = Math.max(1, Math.ceil(weekly.length / maxLabels));
        for (let i = 0; i < weekly.length; i += step) {
            const text = document.createElementNS(ns, 'text');
            text.setAttribute('x', pad + i * stepX);
            text.setAttribute('y', vh + 14);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '9');
            text.setAttribute('fill', 'var(--subtext)');
            text.setAttribute('opacity', '0.7');
            text.textContent = weekDates[i];
            xAxisGroup.appendChild(text);
        }
    }

    function toCoords(pts, min, range, stepX) {
        return pts.map((v, i) => ({
            x: pad + i * stepX,
            y: vh - pad - ((v - min) / range) * (vh - pad * 2),
        }));
    }

    function render() {
        const d = trendData[activeTrendMetric];
        if (!d) return;
        const pts = d.points;
        const hasPlatforms = d.platforms && Object.keys(d.platforms).length > 1;

        // Compute min/max across all series for consistent y-axis (ignore nulls)
        let allValues = [...pts];
        if (hasPlatforms) {
            for (const platPts of Object.values(d.platforms)) allValues = allValues.concat(platPts.filter((p) => p != null));
        }
        const min = Math.min(...allValues) - 5;
        const max = Math.max(...allValues) + 5;
        const range = max - min || 1;
        const stepX = (vw - pad * 2) / (pts.length - 1);

        const coords = toCoords(pts, min, range, stepX);

        // Platform lines
        platformGroup.innerHTML = '';
        if (hasPlatforms) {
            // In multi-line mode: hide aggregate area, show aggregate as thin dashed
            areaEl.setAttribute('points', '');
            lineEl.setAttribute('points', '');
            dotEnd.style.display = 'none';

            for (const [plat, platPts] of Object.entries(d.platforms)) {
                const color = getPlatformColors()[plat]?.stroke || '#666';

                // Break line at null gaps — render continuous segments
                let segment = [];
                for (let j = 0; j < platPts.length; j++) {
                    if (platPts[j] != null) {
                        const x = pad + j * stepX;
                        const y = vh - pad - ((platPts[j] - min) / range) * (vh - pad * 2);
                        segment.push({ x, y });
                    } else if (segment.length > 0) {
                        // Flush segment
                        const line = document.createElementNS(ns, 'polyline');
                        line.setAttribute('points', segment.map((c) => `${c.x},${c.y}`).join(' '));
                        line.setAttribute('fill', 'none');
                        line.setAttribute('stroke', color);
                        line.setAttribute('stroke-width', '2.5');
                        line.setAttribute('stroke-linecap', 'round');
                        line.setAttribute('stroke-linejoin', 'round');
                        platformGroup.appendChild(line);
                        segment = [];
                    }
                }
                // Flush remaining segment
                if (segment.length > 0) {
                    const line = document.createElementNS(ns, 'polyline');
                    line.setAttribute('points', segment.map((c) => `${c.x},${c.y}`).join(' '));
                    line.setAttribute('fill', 'none');
                    line.setAttribute('stroke', color);
                    line.setAttribute('stroke-width', '2.5');
                    line.setAttribute('stroke-linecap', 'round');
                    line.setAttribute('stroke-linejoin', 'round');
                    platformGroup.appendChild(line);

                    // End dot on last non-null point
                    const last = segment[segment.length - 1];
                    const dot = document.createElementNS(ns, 'circle');
                    dot.setAttribute('cx', last.x);
                    dot.setAttribute('cy', last.y);
                    dot.setAttribute('r', '3.5');
                    dot.setAttribute('fill', color);
                    platformGroup.appendChild(dot);
                }
            }
        } else {
            // Single line mode
            dotEnd.style.display = '';
            areaEl.setAttribute('points',
                `${coords[0].x},${vh} ` + coords.map((c) => `${c.x},${c.y}`).join(' ') + ` ${coords[coords.length - 1].x},${vh}`
            );
            lineEl.setAttribute('points', coords.map((c) => `${c.x},${c.y}`).join(' '));

            const last = coords[coords.length - 1];
            dotEnd.setAttribute('cx', last.x);
            dotEnd.setAttribute('cy', last.y);
        }

        dotHover.style.display = 'none';
        dotGlow.style.display = 'none';
        guideEl.style.display = 'none';
        tooltip.style.opacity = '0';

        // Resolve color — platform color for single-platform views, or config override, or accent
        const cfg = TREND_METRIC_CONFIG[activeTrendMetric];
        const rawColor = cfg?.color
            ? getComputedStyle(document.documentElement).getPropertyValue(cfg.color.replace('var(', '').replace(')', '')).trim() || cfg.color
            : null;
        const accent = rawColor || resolveAccent();
        if (!hasPlatforms) {
            lineEl.setAttribute('stroke', accent);
            dotEnd.setAttribute('fill', accent);
        }
        dotHover.setAttribute('fill', accent);
        dotGlow.setAttribute('stroke', hexToRgba(accent, 0.3));

        // Update area gradient to match current accent
        const grad = document.getElementById('areaGrad');
        if (grad) {
            grad.children[0]?.setAttribute('stop-color', hexToRgba(accent, 0.2));
            grad.children[1]?.setAttribute('stop-color', hexToRgba(accent, 0));
        }

        // X-axis
        renderXAxis(stepX);

        // Hit areas for hover
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
                if (!hasPlatforms) {
                    dotHover.setAttribute('cx', c.x);
                    dotHover.setAttribute('cy', c.y);
                    dotGlow.setAttribute('cx', c.x);
                    dotGlow.setAttribute('cy', c.y);
                    dotHover.style.display = '';
                    dotGlow.style.display = '';
                }
                guideEl.setAttribute('x1', c.x);
                guideEl.setAttribute('y1', pad);
                guideEl.setAttribute('x2', c.x);
                guideEl.setAttribute('y2', vh);
                guideEl.style.display = '';
                const svgRect = svgEl.getBoundingClientRect();
                const pctX = c.x / vw;
                let leftPx = pctX * svgRect.width;
                // Clamp tooltip to chart bounds
                const tipW = 180;
                leftPx = Math.max(tipW / 2, Math.min(leftPx, svgRect.width - tipW / 2));
                tooltip.style.left = leftPx + 'px';
                tooltip.style.opacity = '1';

                if (hasPlatforms) {
                    const parts = Object.entries(d.platforms)
                        .filter(([, platPts]) => platPts[i] != null)
                        .map(([plat, platPts]) => {
                            const label = getPlatformColors()[plat]?.label || plat;
                            return `${label}: ${platPts[i]}${d.suffix}`;
                        });
                    tooltip.textContent = parts.length > 0
                        ? `${weekDates[i]} · ${parts.join(' / ')}`
                        : weekDates[i];
                } else {
                    tooltip.textContent = `${weekDates[i]}: ${pts[i]}${d.suffix}`;
                }
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
        const headlineRaw = d.lifetime != null ? Math.round(d.lifetime) : d.points[d.points.length - 1];
        valEl.textContent = (key === 'activity' ? formatCompact(headlineRaw) : headlineRaw) + d.suffix;
        labelEl.textContent = `${d.label} · ${weekly.length}-week trend`;
        // Set headline color to match platform/accent
        const trendPanel = valEl.closest('.trend-panel');
        if (trendPanel) {
            const headlineColor = config?.color
                ? getComputedStyle(document.documentElement).getPropertyValue(config.color.replace('var(', '').replace(')', '')).trim() || config.color
                : resolveAccent();
            trendPanel.style.setProperty('--trend-color', headlineColor);
        }
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
    if (wordsTypedSubtitle) wordsTypedSubtitle.textContent = `${formatCompact(volume.total_words_assistant)} from assistants`;

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

async function openSettings() {
    const modal = document.getElementById('settingsModal');
    if (!modal) return;

    settingsOpener = document.activeElement;

    // Show modal immediately with loading state
    const container = document.getElementById('settingsBackends');
    if (container) container.innerHTML = '<div class="wizard-loading">Loading...</div>';
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Load config (fast — reads local JSON file)
    const configRes = await fetch('/api/config').then(r => r.json()).catch(() => ({}));

    // Populate from config without waiting for detect (no scan needed for settings)
    const backendIds = Object.keys(configRes.backends || {});
    const nameMap = { claude_code: 'Claude Code', codex: 'Codex', copilot_chat: 'Copilot Chat', cursor: 'Cursor' };
    if (container) {
        container.innerHTML = backendIds.map(id => {
            const toggle = configRes.backends[id];
            return `<label class="wizard-toggle">
                <input type="checkbox" data-backend="${id}" ${toggle.enabled !== false ? 'checked' : ''}>
                ${nameMap[id] || id}
            </label>`;
        }).join('');
    }

    // Show exclusions if Claude Code is configured
    const excSection = document.getElementById('settingsExclusionSection');
    if (excSection && configRes.backends?.claude_code) {
        excSection.style.display = 'block';
        const exclusions = configRes.backends.claude_code.exclusions ?? [];
        loadExclusionChips('settingsExclusionChips', exclusions);
    }
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
    const sorted = [...leaderboardData].sort((a, b) => (b[sortKey] ?? 0) - (a[sortKey] ?? 0));
    const myFp = localStorage.getItem('howiprompt_fingerprint');

    body.innerHTML = sorted.map((entry, i) => {
        const isYou = myFp && entry.fingerprint === myFp;
        return `
        <tr${isYou ? ' class="is-you"' : ''}>
            <td class="rank-col">${i + 1}</td>
            <td><strong>${escapeHtml(entry.display_name || 'Anonymous')}</strong>${isYou ? ' <span style="font-size:11px;color:var(--accent)">(you)</span>' : ''}</td>
            <td>${entry.hitl_score ?? '--'}</td>
            <td>${entry.vibe_index ?? '--'}</td>
            <td>${entry.politeness ?? '--'}</td>
            <td>${(entry.total_prompts ?? 0).toLocaleString()}</td>
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
    let name = localStorage.getItem('howiprompt_username');
    if (!name && typeof window.generateRandomUsername === 'function') {
        name = window.generateRandomUsername();
        localStorage.setItem('howiprompt_username', name);
    }
    return name || 'Anonymous';
}

function computeFingerprint() {
    if (!activeView) return null;
    const v = activeView.volume || {};
    const nlp = activeView.nlp || {};
    const raw = `${v.total_human || 0}-${v.total_conversations || 0}-${Math.round(nlp.hitl_score?.avg_score ?? 0)}-${Math.round(nlp.vibe_coder_index?.avg_score ?? 0)}-${Math.round(nlp.politeness?.avg_score ?? 0)}`;
    // Simple hash
    let hash = 0;
    for (let i = 0; i < raw.length; i++) {
        hash = ((hash << 5) - hash + raw.charCodeAt(i)) | 0;
    }
    return 'fp_' + Math.abs(hash).toString(36);
}

// === Submit to leaderboard ===

function getSubmissionPayload() {
    if (!activeView) return null;
    const v = activeView.volume || {};
    const nlp = activeView.nlp || {};
    const persona = activeView.persona || {};

    return {
        hitl_score: Math.round(nlp.hitl_score?.avg_score ?? 0),
        vibe_index: Math.round(nlp.vibe_coder_index?.avg_score ?? 0),
        politeness: Math.round(nlp.politeness?.avg_score ?? 0),
        total_prompts: v.total_human || 0,
        total_conversations: v.total_conversations || 0,
        persona: persona.type || 'unknown',
        platform: activeSourceKey,
    };
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

    const fingerprint = computeFingerprint();
    if (!fingerprint) return;

    payload.display_name = displayName;
    payload.fingerprint = fingerprint;
    payload.updated_at = new Date().toISOString();

    localStorage.setItem('howiprompt_username', displayName);
    localStorage.setItem('howiprompt_fingerprint', fingerprint);
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
        const sorted = [...leaderboardData].sort((a, b) => (b[sortKey] ?? 0) - (a[sortKey] ?? 0));
        const rank = sorted.findIndex(e => e.fingerprint === fingerprint) + 1;

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
    const fp = localStorage.getItem('howiprompt_fingerprint');
    if (!fp) return;
    const toast = document.getElementById('leaderboardToast');
    try {
        const res = await fetch(`${SUPABASE_URL}${LB_TABLE}?fingerprint=eq.${fp}`, {
            method: 'DELETE',
            headers: supaHeaders(false),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        localStorage.removeItem('howiprompt_fingerprint');
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
        wizardBackendData = backends;

        container.innerHTML = backends.map(b => {
            let detail = '';
            if (b.status === 'available' && b.messageCount) {
                detail = `${b.messageCount.toLocaleString()} messages`;
                if (b.dateRange) {
                    const fmt = (iso) => new Date(iso).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
                    detail += ` · ${fmt(b.dateRange.first)} – ${fmt(b.dateRange.last)}`;
                }
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

    const available = wizardBackendData.filter(b => b.status === 'available');
    container.innerHTML = available.map(b => `
        <label class="wizard-toggle">
            <input type="checkbox" data-backend="${b.id}" checked>
            ${b.name}
        </label>
    `).join('');

    // Show exclusions + load existing chips if Claude Code is available
    if (available.some(b => b.id === 'claude_code') && exclusionsDiv) {
        exclusionsDiv.style.display = 'block';
    }
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
    const log = document.getElementById('wizardLog');
    const bar = document.getElementById('wizardProgressBar');
    const doneBtn = document.getElementById('wizardDone');
    if (!log) return;

    log.innerHTML = '<div class="wizard-log-entry">Starting pipeline...</div>';
    const stages = ['sync', 'parse', 'insert', 'nlp', 'embedding', 'classifiers', 'metrics'];
    let maxStageIdx = 0;
    let completed = false;

    const evtSource = new EventSource('/api/pipeline/stream');

    evtSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        const entry = document.createElement('div');
        entry.className = 'wizard-log-entry';
        entry.textContent = `${data.stage}: ${data.detail}`;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;

        const idx = stages.indexOf(data.stage);
        if (idx >= 0 && idx > maxStageIdx) maxStageIdx = idx;
        if (bar) bar.style.width = `${((maxStageIdx + 1) / stages.length) * 100}%`;
    });

    evtSource.addEventListener('complete', (e) => {
        completed = true;
        evtSource.close();
        const data = JSON.parse(e.data);

        const entry = document.createElement('div');
        entry.className = 'wizard-log-entry done';
        entry.textContent = `Done! ${data.stats.totalMessages} messages analyzed.`;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;

        if (bar) bar.style.width = '100%';
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
        const entry = document.createElement('div');
        entry.className = 'wizard-log-entry error';
        entry.textContent = `Error: ${data.message}`;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
        if (doneBtn) {
            doneBtn.textContent = 'Continue Anyway';
            doneBtn.style.display = 'inline-block';
        }
    });

    // Built-in error fires on connection close — ignore if we already completed
    evtSource.onerror = () => {
        if (!completed) {
            evtSource.close();
            const entry = document.createElement('div');
            entry.className = 'wizard-log-entry error';
            entry.textContent = 'Connection lost — check terminal for details.';
            log.appendChild(entry);
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

    // Check if wizard is needed
    const wizardActive = await initWizard();
    if (wizardActive) return;

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
