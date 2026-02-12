// theme.js — Shared theme toggle logic
// Dashboard uses 'dashboard-theme' key, wrapped uses 'theme' key.

export function initThemeToggle(storageKey = 'dashboard-theme') {
    const html = document.documentElement;
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    const saved = localStorage.getItem(storageKey);
    if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        html.classList.add('dark');
    }

    themeToggle.addEventListener('click', () => {
        html.classList.toggle('dark');
        localStorage.setItem(storageKey, html.classList.contains('dark') ? 'dark' : 'light');
    });
}
