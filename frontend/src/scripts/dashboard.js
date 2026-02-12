// dashboard.js — Dashboard interactivity: source filtering, heatmap, trends, launch kit, modal
// Loads metrics.json via fetch() at runtime — no build-time data injection.

import { initThemeToggle } from './theme.js';

let sourceViews = {};
let launchPackets = {};
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

// === Trend rendering ===

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

function renderSourceShareSparkline(trends) {
    const spark = document.getElementById('sourceShareSparkline');
    const summary = document.getElementById('sourceShareSummary');
    if (!spark || !summary) return;
    const daily = (trends && trends.daily_rollups) ? trends.daily_rollups.slice(-30) : [];

    if (daily.length < 2) {
        spark.innerHTML = '';
        summary.textContent = 'Not enough trend data yet';
        return;
    }

    const values = daily.map((row) => Number(row.source_share_pct?.codex ?? 0));
    const points = values.map((value, index) => {
        const x = (index / (values.length - 1)) * 100;
        const y = 30 - (Math.max(0, Math.min(100, value)) / 100) * 30;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    spark.innerHTML = `<path d="M${points.join(' L')}" />`;
    const first = values[0];
    const last = values[values.length - 1];
    const delta = Math.round((last - first) * 10) / 10;
    const trendWord = delta > 0 ? 'up' : (delta < 0 ? 'down' : 'flat');
    summary.textContent = `Codex share is ${trendWord} (${delta > 0 ? '+' : ''}${delta} pts, 30d window)`;
}

function renderStyleTrend(deltas) {
    const container = document.getElementById('styleTrendList');
    if (!container) return;
    container.innerHTML = '';
    const rows = [
        ['Question Rate', deltas?.question_rate_pct],
        ['Command Rate', deltas?.command_rate_pct],
        ['Politeness', deltas?.politeness_per_100],
        ['Backtrack', deltas?.backtrack_per_100],
    ];

    rows.forEach(([label, stats]) => {
        const delta = formatDelta(stats?.delta_pct);
        const row = document.createElement('div');
        row.className = 'trend-row';
        row.innerHTML = `
            <span><strong>${label}</strong> <span class="trend-note">7d: ${stats?.avg_7d ?? 'n/a'} · 30d: ${stats?.avg_30d ?? 'n/a'}</span></span>
            <span class="trend-delta ${delta.cls}">${delta.text}</span>
        `;
        container.appendChild(row);
    });
}

function renderModelUsage(modelUsage) {
    const container = document.getElementById('modelUsageList');
    if (!container) return;
    container.innerHTML = '';
    const byModel = modelUsage?.by_model || [];
    const topModels = byModel.slice(0, 4);
    if (topModels.length === 0) {
        container.innerHTML = '<div class="trend-note">No model metadata available for this view</div>';
        return;
    }

    topModels.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'trend-row';
        row.innerHTML = `
            <span><strong>${item.model_id}</strong> <span class="trend-note">${item.model_provider}</span></span>
            <span>${item.prompts} prompts</span>
        `;
        container.appendChild(row);
    });

    const coverage = modelUsage?.coverage?.metadata_coverage_pct ?? 0;
    const coverageNote = document.createElement('div');
    coverageNote.className = 'trend-note';
    coverageNote.textContent = `Metadata coverage: ${coverage}% of prompts`;
    container.appendChild(coverageNote);
}

function renderTrendCallouts(trends, deltas) {
    const container = document.getElementById('trendCalloutsList');
    if (!container) return;
    container.innerHTML = '';
    const notes = [];

    const promptsDelta = formatDelta(deltas?.prompts_per_day?.delta_pct);
    notes.push(`Prompt volume vs baseline: ${promptsDelta.text} (7d vs 30d)`);

    const codexDelta = formatDelta(deltas?.codex_share_pct?.delta_pct);
    notes.push(`Codex share trend: ${codexDelta.text} (7d vs 30d)`);

    const shifts = (trends?.shift_markers || []).slice(0, 3);
    shifts.forEach((marker) => {
        if (marker.type === 'prompt_shift') {
            notes.push(`${marker.date}: prompt volume ${marker.direction} by ${marker.delta_prompts}`);
        } else if (marker.type === 'source_share_shift') {
            notes.push(`${marker.date}: Codex share ${marker.direction} ${marker.delta_codex_share_pct} pts`);
        }
    });

    notes.forEach((note) => {
        const row = document.createElement('div');
        row.className = 'trend-note';
        row.textContent = `• ${note}`;
        container.appendChild(row);
    });
}

function renderTrendBand(view) {
    const trends = view?.trends || {};
    const deltas = trends?.deltas_7d_vs_30d || {};
    renderSourceShareSparkline(trends);
    renderStyleTrend(deltas);
    renderModelUsage(view?.model_usage || {});
    renderTrendCallouts(trends, deltas);
}

// === Launch kit ===

function getLaunchPacket(sourceKey) {
    return launchPackets[sourceKey] || launchPackets.both || null;
}

async function copyTextToClipboard(text) {
    if (!text) return false;
    if (navigator?.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (_) {
            // Fall back to legacy copy path.
        }
    }

    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'absolute';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    document.body.removeChild(textarea);
    return copied;
}

async function copyLaunchText(kind) {
    const status = document.getElementById('launchCopyStatus');
    const packet = getLaunchPacket(activeSourceKey);
    if (!packet || !packet[kind]) {
        if (status) status.textContent = 'No launch text available for this source view yet.';
        return;
    }

    const copied = await copyTextToClipboard(packet[kind]);
    if (status) {
        status.textContent = copied
            ? `Copied ${kind.replace('_', ' ')} for ${packet.source_label}.`
            : 'Copy failed. Please copy manually from the preview.';
    }
}

function renderLaunchKit(sourceKey, view) {
    const preview = document.getElementById('launchSummaryPreview');
    const status = document.getElementById('launchCopyStatus');
    const packet = getLaunchPacket(sourceKey);

    if (!packet) {
        if (preview) preview.textContent = 'Launch packet unavailable for this source view.';
        if (status) status.textContent = 'No share text available.';
        return;
    }

    const prompts = view?.volume?.total_human ?? 0;
    if (preview) preview.textContent = packet.summary;
    if (status) status.textContent = `${packet.source_label} ready: ${Number(prompts).toLocaleString()} prompts included with GitHub attribution.`;
}

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

// === Launch kit button handlers ===

document.getElementById('copyLaunchSummaryBtn')?.addEventListener('click', () => copyLaunchText('summary'));
document.getElementById('copyReleaseNotesBtn')?.addEventListener('click', () => copyLaunchText('release_notes'));
document.getElementById('copyHnPostBtn')?.addEventListener('click', () => copyLaunchText('hn_post'));
document.getElementById('copyLinkedinPostBtn')?.addEventListener('click', () => copyLaunchText('linkedin_post'));

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
    const launchGithubLink = document.getElementById('launchGithubLink');
    if (launchGithubLink) launchGithubLink.href = githubRepo;
}

// === Init ===

async function init() {
    initThemeToggle('dashboard-theme');
    fixLocalLinks();

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

        launchPackets = metrics.launch_packets || {};
        applyBranding(metrics);
        initSourceFilter();
    } catch (err) {
        console.warn('Could not load metrics.json:', err.message);
        // Page remains with placeholder values
    }
}

init();
