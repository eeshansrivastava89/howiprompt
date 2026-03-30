// wrapped.js — Wrapped experience interactivity
// Source filtering, heatmap, counter animations, embedding metrics, persona, share card.
// Loads metrics.json via fetch() at runtime.

import { initThemeToggle } from './theme.js';
import { SOURCE_LABELS, SOURCE_ACCENTS, formatSourceLabel, getSourceDisplayName, formatHour12, formatDateRange } from './shared.js';

let sourceViews = {};
let activeSourceKey = 'both';
const sourceBar = document.getElementById('sourceBar');

let settingsOpener = null;
let detectedBackendCache = null;
let detectedBackendFetch = null;

function getCachedDetectedBackends() {
    return Array.isArray(detectedBackendCache) ? detectedBackendCache : [];
}

function setDetectedBackends(backends) {
    detectedBackendCache = Array.isArray(backends) ? backends : [];
    return detectedBackendCache;
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
            ? 'Checking installed backends...'
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

async function pickAndAddExclusion(chipContainerId) {
    const container = document.getElementById(chipContainerId);
    if (!container) return;

    let dirPath;
    try {
        const res = await fetch('/api/pick-directory');
        const data = await res.json();
        dirPath = data.path;
    } catch {
        return;
    }
    if (!dirPath) return;
    if (container.querySelector(`[data-path="${CSS.escape(dirPath)}"]`)) return;

    let messageCount = 0;
    try {
        const res = await fetch('/api/exclusion-count', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: dirPath }),
        });
        const data = await res.json();
        messageCount = data.messageCount || 0;
    } catch {}

    addExclusionChip(container, dirPath, messageCount);
}

function addExclusionChip(container, dirPath, messageCount) {
    const chip = document.createElement('div');
    chip.className = 'exclusion-chip';
    chip.dataset.path = dirPath;
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
    return [...container.querySelectorAll('.exclusion-chip')].map((chip) => chip.dataset.path);
}

async function loadExclusionChips(chipContainerId, exclusions) {
    const container = document.getElementById(chipContainerId);
    if (!container) return;
    container.innerHTML = '';
    for (const dirPath of exclusions) {
        let messageCount = 0;
        try {
            const res = await fetch('/api/exclusion-count', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: dirPath }),
            });
            const data = await res.json();
            messageCount = data.messageCount || 0;
        } catch {}
        addExclusionChip(container, dirPath, messageCount);
    }
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

    const excSection = document.getElementById('settingsExclusionSection');
    if (excSection && configRes.backends?.claude_code) {
        excSection.style.display = 'block';
        const exclusions = configRes.backends.claude_code.exclusions ?? [];
        loadExclusionChips('settingsExclusionChips', exclusions);
    } else if (excSection) {
        excSection.style.display = 'none';
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
        if (settingsOpener) {
            settingsOpener.focus();
            settingsOpener = null;
        }
    }
}

function startRefreshModal() {
    const modal = document.getElementById('refreshModal');
    const log = document.getElementById('refreshLog');
    const bar = document.getElementById('refreshProgressBar');
    const resultEl = document.getElementById('refreshModalResult');
    const closeBtn = document.getElementById('refreshModalClose');
    if (!modal || !log || !bar || !resultEl || !closeBtn) return null;

    log.innerHTML = '<div class="wizard-log-entry">Starting pipeline...</div>';
    bar.style.width = '0';
    resultEl.style.display = 'none';
    closeBtn.style.display = 'none';
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    return { modal, log, bar, resultEl, closeBtn };
}

async function handleRefresh() {
    const ui = startRefreshModal();
    if (!ui) {
        window.location.reload();
        return;
    }

    const { log, bar, resultEl, closeBtn } = ui;
    const stages = ['sync', 'parse', 'insert', 'nlp', 'embedding', 'classifiers', 'metrics'];
    let maxStageIdx = 0;
    let refreshDone = false;
    const evtSource = new EventSource('/api/pipeline/stream');

    const showResult = (html) => {
        resultEl.innerHTML = html;
        resultEl.style.display = 'block';
        closeBtn.style.display = 'inline-block';
        closeBtn.focus();
    };

    evtSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        const entry = document.createElement('div');
        entry.className = 'wizard-log-entry';
        entry.textContent = `${data.stage}: ${data.detail}`;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
        const idx = stages.indexOf(data.stage);
        if (idx >= 0 && idx > maxStageIdx) maxStageIdx = idx;
        bar.style.width = `${((maxStageIdx + 1) / stages.length) * 100}%`;
    });

    evtSource.addEventListener('complete', (e) => {
        refreshDone = true;
        evtSource.close();
        const data = JSON.parse(e.data);
        bar.style.width = '100%';
        const entry = document.createElement('div');
        entry.className = 'wizard-log-entry done';
        entry.textContent = `Done! ${data.stats.totalMessages.toLocaleString()} messages analyzed.`;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;

        if (data.metrics) {
            sourceViews = data.metrics.source_views || { both: data.metrics };
            if (!sourceViews.claude_code && sourceViews.claude) sourceViews.claude_code = sourceViews.claude;
            sourceViews.both = sourceViews.both || data.metrics;
            if (!sourceViews[activeSourceKey]) activeSourceKey = 'both';
            initSourceFilter();
        }

        showResult('Re-analysis complete. Wrapped has been updated with the latest metrics.');
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
            showResult('Refresh failed -- is the server running?');
        }
    };
}

async function saveSettings() {
    const checked = document.querySelectorAll('#settingsBackends input[type=checkbox]:checked');
    if (checked.length === 0) {
        alert('Please enable at least one backend to analyze.');
        return;
    }

    const toggles = {};
    document.querySelectorAll('#settingsBackends input[type=checkbox]').forEach((cb) => {
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

// === Heatmap ===

function renderHeatmap(heatmapData) {
    const data = Array.isArray(heatmapData) ? heatmapData : [];
    const heatmapContainer = document.getElementById('heatmap');
    if (!heatmapContainer || data.length === 0) return;

    const isMobile = window.innerWidth < 640;
    const days = isMobile ? ['M', 'T', 'W', 'T', 'F', 'S', 'S'] : ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const maxVal = Math.max(1, ...data.flat());

    heatmapContainer.innerHTML = '';
    data.forEach((row, dayIndex) => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'flex items-center gap-[1px] sm:gap-[2px]';
        const dayLabel = document.createElement('span');
        dayLabel.className = (isMobile ? 'w-8' : 'w-12') + ' text-xs text-muted shrink-0';
        dayLabel.textContent = days[dayIndex];
        rowDiv.appendChild(dayLabel);
        const cellsContainer = document.createElement('div');
        cellsContainer.className = 'flex-1 grid grid-cols-24 gap-[1px] sm:gap-[2px]';
        row.forEach((value, hourIndex) => {
            const cell = document.createElement('div');
            const intensity = value > 0 ? Math.ceil((value / maxVal) * 5) : 0;
            const opacityMap = { 0: '0.1', 1: '0.2', 2: '0.35', 3: '0.55', 4: '0.75', 5: '1' };
            cell.className = 'heatmap-cell aspect-square rounded-sm cursor-pointer';
            cell.style.background = intensity > 0 ? `rgba(var(--accent-rgb), ${opacityMap[intensity]})` : 'var(--border)';
            if (intensity === 0) cell.style.opacity = '0.3';
            cell.title = `${days[dayIndex]} ${formatHour12(hourIndex)} — ${value} prompt${value !== 1 ? 's' : ''}`;
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

// === Donut + Arc helpers ===

const DONUT_CIRCUMFERENCE = 125.66;
const ARC_CIRCUMFERENCE = 326.73;

function setDonut(circleId, valId, score) {
    const circle = document.getElementById(circleId);
    const valEl = document.getElementById(valId);
    const value = Math.round(score ?? 0);
    if (circle) {
        const offset = DONUT_CIRCUMFERENCE * (1 - Math.min(value, 100) / 100);
        circle.setAttribute('stroke-dashoffset', String(offset));
    }
    if (valEl) valEl.textContent = value;
}

function setArc(arcId, score) {
    const arc = document.getElementById(arcId);
    if (arc) {
        const pct = Math.min(score ?? 0, 100) / 100;
        arc.setAttribute('stroke-dashoffset', String(ARC_CIRCUMFERENCE * (1 - pct)));
    }
}

// === Fix dashboard link for local vs production ===

function fixLocalLinks() {
    const isLocal = location.protocol === 'file:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (isLocal) {
        document.querySelectorAll('.dashboard-link').forEach(link => {
            const currentPath = location.pathname;
            if (currentPath.includes('/wrapped')) {
                const wrappedIdx = currentPath.indexOf('/wrapped');
                link.href = currentPath.substring(0, wrappedIdx) + '/index.html';
            } else {
                link.href = '/index.html';
            }
        });
    }
}

// === Hydrate all wrapped sections ===

function hydrateWrapped(m) {
    const v = m.volume || {};
    const t = m.temporal || {};
    const yr = m.youre_right || {};
    const cd = m.conversation_depth || {};
    const dr = m.date_range || {};
    const nlp = m.nlp || {};
    const p = m.persona || {};

    const el = (id) => document.getElementById(id);
    const setText = (id, text) => { const e = el(id); if (e) e.textContent = text; };

    const dateRangeDisplay = formatDateRange(dr);
    const peakHour12h = formatHour12(t.peak_hour);

    // Section 1: Cold Open
    setText('yrCount', yr.count || 0);
    setText('yrPerConvo', `${yr.per_conversation || 0}x`);
    setText('dateRangeDisplay', dateRangeDisplay);
    setText('dateRange', dateRangeDisplay);

    // Platform-aware attribution
    const ps = m.platform_stats || {};
    const platformNames = Object.keys(ps).map(k => getSourceDisplayName(k, true));
    const attribution = activeSourceKey !== 'both'
        ? getSourceDisplayName(activeSourceKey, true)
        : (platformNames.length > 0 ? platformNames.join(' & ') : 'the AI');
    setText('yrAttribution', attribution);

    // Section 2: Numbers
    const counterEl = el('mainCounter');
    if (counterEl) {
        counterEl.dataset.target = v.total_human || 0;
        counterEl.textContent = (v.total_human || 0).toLocaleString();
    }
    setText('totalConversations', (v.total_conversations || 0).toLocaleString());
    setText('totalWordsK', `${Math.round((v.total_words_human || 0) / 1000)}K`);
    setText('avgTurns', cd.avg_turns || 0);
    setText('maxTurns', cd.max_turns || 0);
    setText('deepDives', cd.deep_dives || 0);

    // Section 3: Temporal
    setText('peakHour', peakHour12h);
    setText('peakHourCount', `${t.peak_hour_count || 0} prompts`);
    setText('nightOwlPct', `${t.night_owl_pct || 0}%`);
    setText('peakDay', t.peak_day || 'N/A');
    setText('peakDayCount', `${t.peak_day_count || 0} prompts`);

    // Section 4: Your Style (embedding hero metrics)
    const vibeRaw = nlp.vibe_coder_index?.avg_score;
    const vibeScore = vibeRaw != null ? 100 - vibeRaw : null;
    const politeScore = nlp.politeness?.avg_score;

    setText('wrappedVibeValue', vibeScore != null ? Math.round(vibeScore) : '--');
    setText('wrappedVibeLabel', vibeScore >= 50 ? 'Vibe Coder' : vibeScore != null ? 'Engineer' : 'No data');
    const vibeMarker = el('wrappedVibeMarker');
    if (vibeMarker && vibeScore != null) vibeMarker.style.left = `${Math.min(Math.max(vibeScore, 0), 100)}%`;

    setText('wrappedPoliteValue', politeScore != null ? Math.round(politeScore) : '--');
    setText('wrappedPoliteLabel', politeScore >= 66 ? 'Courteous' : politeScore >= 33 ? 'Balanced' : politeScore != null ? 'Direct' : 'No data');
    setArc('wrappedPoliteArc', politeScore);

    // Dynamic style headline
    if (vibeScore != null && politeScore != null) {
        const v = Math.round(vibeScore);
        const p = Math.round(politeScore);
        let headline, subline;
        if (v >= 50 && p >= 50) {
            headline = 'Polite vibes only.';
            subline = `You vibe-code with a ${v} and say please with a ${p}. Warm and loose — the AI loves working with you.`;
        } else if (v >= 50 && p < 50) {
            headline = 'Vibes, no filter.';
            subline = `A ${v} on the vibe scale with a ${p} on politeness. You trust the AI with intent and skip the pleasantries.`;
        } else if (v < 50 && p >= 50) {
            headline = 'Spec-driven, still kind.';
            subline = `A ${v} vibe score means you write specs, not wishes. But a ${p} politeness says you do it with warmth.`;
        } else {
            headline = 'All business.';
            subline = `Vibe score ${v}, politeness ${p}. You're here to ship, not chat. Specs in, code out.`;
        }
        setText('styleHeadline', headline);
        setText('styleSubline', subline);
    }

    // Section 5: Persona Reveal (2×2 quadrant system)
    const QUADRANT_IMAGES = {
        'The Architect': '/images/char_architect.png',
        'The Explorer': '/images/char_explorer.png',
        'The Commander': '/images/char_commander.png',
        'The Partner': '/images/char_partner.png',
    };

    const cardTop = el('wrappedCardTop');
    if (cardTop) cardTop.src = QUADRANT_IMAGES[p.name] || QUADRANT_IMAGES['The Architect'];

    setText('personaName', p.name || '--');
    setText('personaDesc', p.description || '');

    const traitsContainer = el('personaTraits');
    if (traitsContainer) {
        traitsContainer.innerHTML = '';
        (p.traits || []).forEach(trait => {
            const span = document.createElement('span');
            span.className = 'px-4 py-1.5 text-sm font-medium rounded-full';
            span.style.background = 'var(--accent-soft)';
            span.textContent = trait;
            traitsContainer.appendChild(span);
        });
    }

    setDonut('wDonutDetail', 'wValDetail', p.detail_score);
    setDonut('wDonutStyle', 'wValStyle', p.style_score);

    // Section 6: Terminal Summary
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
    setText('termVibe', vibeScore != null ? Math.round(vibeScore) : '--');  // already inverted above
    setText('termPolite', politeScore != null ? Math.round(politeScore) : '--');
    setText('termYrCount', `${yr.count || 0}x`);

    // Platform breakdown in receipt
    const platSection = el('termPlatformSection');
    if (platSection) {
        const ps = m.platform_stats || {};
        platSection.innerHTML = '';
        const platforms = Object.keys(ps);
        if (platforms.length > 0) {
            platforms.forEach(plat => {
                const stats = ps[plat];
                const row = document.createElement('div');
                row.className = 'flex justify-between';
                row.innerHTML = `<span class="text-muted">${formatSourceLabel(plat)}</span><span class="font-bold">${(stats.messages || 0).toLocaleString()} msgs</span>`;
                platSection.appendChild(row);
            });
        }
    }
    setText('termPersonaName', `PERSONA: ${(p.name || '--').toUpperCase()}`);
    setText('termPersonaTraits', (p.traits || []).join(' · '));

    // Render heatmap
    renderHeatmap(t.heatmap);

    // Apply branding links
    const branding = m.branding || {};
    if (branding.site_url) {
        const brandingLink = el('brandingLink');
        if (brandingLink) brandingLink.href = branding.site_url;
    }
    if (branding.site_name) setText('brandingSiteName', branding.site_name);
    if (branding.github_repo) {
        document.querySelectorAll('.build-your-own-link').forEach(link => {
            link.href = branding.github_repo;
        });
    }
}

// === Source filter ===

function initSourceFilter() {
    if (!sourceBar) return;
    const available = Object.keys(sourceViews)
        .filter(k => k === 'both' || Boolean(sourceViews[k]))
        .sort((a, b) => {
            if (a === 'both') return 1;
            if (b === 'both') return -1;
            return formatSourceLabel(a).localeCompare(formatSourceLabel(b));
        });

    sourceBar.innerHTML = '';
    for (const key of available) {
        const btn = document.createElement('button');
        btn.className = 'source-pill';
        btn.dataset.source = key;
        const color = SOURCE_ACCENTS[key];
        if (color) {
            const dot = document.createElement('span');
            dot.className = 'source-pill-dot';
            dot.style.background = color;
            btn.appendChild(dot);
        }
        btn.appendChild(document.createTextNode(formatSourceLabel(key)));
        sourceBar.appendChild(btn);
    }

    let selected = localStorage.getItem('wrapped-source-filter') || '';
    if (selected === 'both' || !available.includes(selected)) {
        selected = available.find(k => k !== 'both') || available[0] || 'both';
    }

    function selectSource(key) {
        activeSourceKey = key;
        sourceBar.querySelectorAll('.source-pill').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.source === key);
        });
        localStorage.setItem('wrapped-source-filter', key);
        hydrateWrapped(sourceViews[key]);
    }

    sourceBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.source-pill');
        if (btn) selectSource(btn.dataset.source);
    });

    selectSource(selected);
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
document.getElementById('refreshModalClose')?.addEventListener('click', () => {
    const modal = document.getElementById('refreshModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
});

document.getElementById('methodologyModal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeMethodology();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeMethodology();
        closeSettings();
    }
    const modal = document.getElementById('settingsModal')?.classList.contains('active')
        ? document.getElementById('settingsModal')
        : document.getElementById('methodologyModal');
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

// === Init ===

async function init() {
    initThemeToggle();
    fixLocalLinks();
    initAnimations();

    try {
        const response = await fetch('./metrics.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const metricsData = await response.json();
        sourceViews = metricsData.source_views || {};

        if (Object.keys(sourceViews).length === 0) {
            sourceViews = { both: metricsData };
        }

        initSourceFilter();
    } catch (err) {
        console.warn('Could not load metrics.json:', err.message);
    }
}

init();
