// shared.js — Constants and pure utility functions shared by dashboard.js and wrapped.js

export const SOURCE_LABELS = {
    both: 'All',
    claude_code: 'Claude Code',
    codex: 'Codex',
    copilot_chat: 'Copilot Chat',
    cursor: 'Cursor',
    lmstudio: 'LM Studio',
};

export const SOURCE_ACCENTS = {
    claude_code: '#e67e22',
    codex: '#a855f7',
    copilot_chat: '#06b6d4',
    cursor: '#3b82f6',
    lmstudio: '#22c55e',
};

export function formatSourceLabel(key) {
    return SOURCE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function getSourceDisplayName(key, short = false) {
    if (short) {
        if (key === 'claude_code') return 'Claude';
        if (key === 'copilot_chat') return 'Copilot';
    }
    return formatSourceLabel(key);
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
