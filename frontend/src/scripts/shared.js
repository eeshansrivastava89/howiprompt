// shared.js — Constants and pure utility functions shared by dashboard.js and wrapped.js

export const SOURCE_LABELS = {
    both: 'All',
    claude_code: 'Claude Code',
    codex: 'Codex',
    copilot_chat: 'Copilot Chat',
    cursor: 'Cursor',
    lmstudio: 'LM Studio',
    pi: 'Pi',
    opencode: 'OpenCode',
};

export const SOURCE_KEYS = [
    'both',
    'claude_code',
    'codex',
    'copilot_chat',
    'cursor',
    'lmstudio',
    'pi',
    'opencode',
];

export const SOURCE_ACCENTS = {
    claude_code: '#e67e22',
    codex: '#a855f7',
    copilot_chat: '#06b6d4',
    cursor: '#3b82f6',
    lmstudio: '#22c55e',
    pi: '#f43f5e',
    opencode: '#0ea5e9',
};

export function formatSourceLabel(key) {
    return SOURCE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function getSourceDisplayName(key, short = false) {
    if (short) {
        if (key === 'both') return 'All';
        if (key === 'claude_code') return 'Claude';
        if (key === 'copilot_chat') return 'Copilot';
    }
    return formatSourceLabel(key);
}

export function sourceDisabledReason(info) {
    if (!info) return '';
    if (info.supported === false || info.status === 'coming_soon') return 'Detected, but analysis support is not shipped yet';
    if (info.status === 'not_found' || info.detected === false) return 'Supported, but not detected on this machine';
    return '';
}

export function createPillGroup({ container, items, selected, className = 'filter-pills', onSelect }) {
    if (!container) return null;
    container.className = className;
    let currentKey = selected;

    function render() {
        container.innerHTML = '';
        for (const item of items) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = `filter-pill${item.key === currentKey ? ' active' : ''}${item.disabled ? ' is-disabled' : ''}`;
            btn.dataset.key = item.key;
            btn.setAttribute('aria-pressed', item.key === currentKey ? 'true' : 'false');
            if (item.disabled) {
                btn.setAttribute('aria-disabled', 'true');
                if (item.reason) btn.dataset.tooltip = item.reason;
            }
            if (item.color) {
                const dot = document.createElement('span');
                dot.className = 'filter-pill-dot';
                dot.style.background = item.color;
                btn.appendChild(dot);
            }
            const label = document.createElement('span');
            label.textContent = item.label;
            btn.appendChild(label);
            btn.addEventListener('click', () => {
                if (item.disabled) return;
                select(item.key);
            });
            container.appendChild(btn);
        }
    }

    function select(key) {
        const item = items.find((i) => i.key === key);
        if (!item || item.disabled) return;
        currentKey = key;
        render();
        onSelect(key);
    }

    render();
    return { select };
}

export function formatHour12(hour) {
    const h = Number(hour) || 0;
    return `${h % 12 || 12}${h < 12 ? 'am' : 'pm'}`;
}

export function formatDateRange(dateRange) {
    if (!dateRange || !dateRange.first || !dateRange.last) return String(new Date().getFullYear());
    const first = new Date(dateRange.first);
    const last = new Date(dateRange.last);
    const opts = { month: 'short', day: '2-digit', year: 'numeric' };
    return `${first.toLocaleDateString('en-US', opts)} – ${last.toLocaleDateString('en-US', opts)}`;
}

// ── Custom dropdown component ───────────────────────

const CHEVRON_SVG = `<svg class="hip-dd-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>`;

/**
 * Create a styled dropdown that replaces pill rows.
 * @param {Object} opts
 * @param {HTMLElement} opts.container - Element to render into
 * @param {Array<{key:string, label:string, color?:string, desc?:string}>} opts.items
 * @param {string} opts.selected - Initial selected key
 * @param {string} [opts.placeholder] - Label shown before the selected item name
 * @param {(key:string)=>void} opts.onSelect - Callback when selection changes
 * @returns {{select:(key:string)=>void, update:(items:Array)=>void}}
 */
export function createDropdown({ container, items, selected, placeholder, onSelect }) {
    container.classList.add('hip-dd');
    container.innerHTML = '';

    // Trigger button
    const trigger = document.createElement('button');
    trigger.className = 'hip-dd-trigger';
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');
    container.appendChild(trigger);

    // Panel
    const panel = document.createElement('div');
    panel.className = 'hip-dd-panel';
    panel.setAttribute('role', 'listbox');
    container.appendChild(panel);

    let currentKey = selected;
    let currentItems = items;

    function renderTrigger() {
        const item = currentItems.find(i => i.key === currentKey) || currentItems[0];
        if (!item) { trigger.innerHTML = placeholder || ''; return; }
        const dot = item.color ? `<span class="hip-dd-dot" style="background:${item.color}"></span>` : '';
        const pre = placeholder ? `<span class="hip-dd-ph">${placeholder}</span>` : '';
        trigger.innerHTML = `${pre}${dot}<span class="hip-dd-val">${item.label}</span>${CHEVRON_SVG}`;
    }

    function renderPanel() {
        panel.innerHTML = currentItems.map(item => {
            const dot = item.color ? `<span class="hip-dd-dot" style="background:${item.color}"></span>` : '';
            const check = item.key === currentKey ? `<svg class="hip-dd-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>` : '<span class="hip-dd-check-space"></span>';
            const desc = item.desc ? `<span class="hip-dd-item-desc">${item.desc}</span>` : '';
            return `<button class="hip-dd-item ${item.key === currentKey ? 'active' : ''}" data-key="${item.key}" role="option" aria-selected="${item.key === currentKey}">${check}${dot}<span class="hip-dd-item-label">${item.label}${desc}</span></button>`;
        }).join('');
    }

    function open() {
        panel.classList.add('open');
        trigger.setAttribute('aria-expanded', 'true');
        trigger.classList.add('open');
    }

    function close() {
        panel.classList.remove('open');
        trigger.setAttribute('aria-expanded', 'false');
        trigger.classList.remove('open');
    }

    function select(key) {
        currentKey = key;
        renderTrigger();
        renderPanel();
        close();
        onSelect(key);
    }

    trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.classList.contains('open') ? close() : open();
    });

    panel.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-key]');
        if (btn) select(btn.dataset.key);
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
        if (!container.contains(e.target)) close();
    });

    // Close on Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && panel.classList.contains('open')) {
            close();
            trigger.focus();
        }
    });

    renderTrigger();
    renderPanel();

    return {
        select(key) { select(key); },
        update(newItems) {
            currentItems = newItems;
            if (!newItems.find(i => i.key === currentKey)) currentKey = newItems[0]?.key;
            renderTrigger();
            renderPanel();
        },
    };
}
