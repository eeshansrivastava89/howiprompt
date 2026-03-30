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

// Define globally so Header's onclick never throws even if modal is absent
window.openCreateModal = () => {};
window.closeCreateModal = () => {};

function initCreateDrawer() {
    const modal = document.getElementById('createModal');
    if (!modal) return;

    const copyTimers = new WeakMap();
    let opener = null;

    window.openCreateModal = () => {
        opener = document.activeElement;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        const closeBtn = modal.querySelector('.modal-close');
        if (closeBtn) closeBtn.focus();
    };

    window.closeCreateModal = () => {
        modal.classList.remove('active');
        document.body.style.overflow = '';
        if (opener && typeof opener.focus === 'function') {
            opener.focus();
            opener = null;
        }
    };

    const copyButtons = [...document.querySelectorAll('[data-copy-text]')];
    copyButtons.forEach((copyBtn) => {
        const copyLabel = copyBtn.querySelector('.create-copy-label');
        copyBtn.addEventListener('click', async () => {
            const text = copyBtn.dataset.copyText || '';
            try {
                await copyTextWithFallback(text);
                if (copyLabel) copyLabel.textContent = 'Copied';
                copyBtn.classList.add('is-copied');
                const existingTimer = copyTimers.get(copyBtn);
                if (existingTimer) clearTimeout(existingTimer);
                const timer = window.setTimeout(() => {
                    if (copyLabel) copyLabel.textContent = 'Copy';
                    copyBtn.classList.remove('is-copied');
                }, 1600);
                copyTimers.set(copyBtn, timer);
            } catch {
                if (copyLabel) copyLabel.textContent = 'Failed';
                const existingTimer = copyTimers.get(copyBtn);
                if (existingTimer) clearTimeout(existingTimer);
                const timer = window.setTimeout(() => {
                    if (copyLabel) copyLabel.textContent = 'Copy';
                }, 1600);
                copyTimers.set(copyBtn, timer);
            }
        });
    });

    modal.addEventListener('click', (event) => {
        if (event.target === modal) window.closeCreateModal();
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && modal.classList.contains('active')) {
            window.closeCreateModal();
        }
        if (event.key === 'Tab' && modal.classList.contains('active')) {
            const focusable = modal.querySelectorAll('button, [href], [tabindex]:not([tabindex="-1"])');
            if (focusable.length === 0) return;
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault(); last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault(); first.focus();
            }
        }
    });
}

initCreateDrawer();
