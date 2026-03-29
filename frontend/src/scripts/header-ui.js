function copyTextWithFallback(text) {
    if (navigator.clipboard?.writeText) {
        return navigator.clipboard.writeText(text);
    }

    return new Promise((resolve, reject) => {
        try {
            const input = document.createElement('textarea');
            input.value = text;
            input.setAttribute('readonly', '');
            input.style.position = 'absolute';
            input.style.left = '-9999px';
            document.body.appendChild(input);
            input.select();
            document.execCommand('copy');
            document.body.removeChild(input);
            resolve();
        } catch (error) {
            reject(error);
        }
    });
}

function initCreateDrawer() {
    const dropdown = document.getElementById('createDropdown');
    const toggle = document.getElementById('createToggle');
    const menu = document.getElementById('createMenu');
    const copyBtn = document.getElementById('createCopyBtn');
    const copyLabel = copyBtn?.querySelector('.create-copy-label');
    if (!dropdown || !toggle || !menu || !copyBtn) return;

    let copyTimer = null;

    function setOpen(isOpen) {
        dropdown.classList.toggle('open', isOpen);
        toggle.setAttribute('aria-expanded', String(isOpen));
        menu.hidden = !isOpen;
    }

    toggle.addEventListener('click', (event) => {
        event.stopPropagation();
        setOpen(menu.hidden);
    });

    menu.addEventListener('click', (event) => {
        event.stopPropagation();
    });

    copyBtn.addEventListener('click', async () => {
        const text = copyBtn.dataset.copyText || '';
        try {
            await copyTextWithFallback(text);
            if (copyLabel) copyLabel.textContent = 'Copied';
            copyBtn.classList.add('is-copied');
            if (copyTimer) clearTimeout(copyTimer);
            copyTimer = window.setTimeout(() => {
                if (copyLabel) copyLabel.textContent = 'Copy';
                copyBtn.classList.remove('is-copied');
            }, 1600);
        } catch {
            if (copyLabel) copyLabel.textContent = 'Failed';
            if (copyTimer) clearTimeout(copyTimer);
            copyTimer = window.setTimeout(() => {
                if (copyLabel) copyLabel.textContent = 'Copy';
            }, 1600);
        }
    });

    document.addEventListener('click', () => {
        setOpen(false);
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') setOpen(false);
    });
}

initCreateDrawer();
