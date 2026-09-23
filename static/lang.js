(function () {
  const defaultLang = 'zh';

  function applyLang(lang) {
    const normalized = lang === 'en' ? 'en' : 'zh';
    document.documentElement.setAttribute('data-lang', normalized);
    document.documentElement.setAttribute('lang', normalized === 'zh' ? 'zh-CN' : 'en');
    try { localStorage.setItem('lang', normalized); } catch (error) {}

    const titleEl = document.querySelector('title[data-title-zh]');
    if (titleEl) {
      document.title = normalized === 'zh'
        ? titleEl.getAttribute('data-title-zh')
        : titleEl.getAttribute('data-title-en');
    }

    document.querySelectorAll('[data-i18n-placeholder-zh]').forEach((el) => {
      const value = normalized === 'zh'
        ? el.getAttribute('data-i18n-placeholder-zh')
        : el.getAttribute('data-i18n-placeholder-en');
      if (value) {
        el.setAttribute('placeholder', value);
      }
    });

    document.querySelectorAll('[data-i18n-alt-zh]').forEach((el) => {
      const value = normalized === 'zh'
        ? el.getAttribute('data-i18n-alt-zh')
        : el.getAttribute('data-i18n-alt-en');
      if (value) {
        el.setAttribute('alt', value);
      }
    });

    document.querySelectorAll('[data-i18n-text-zh]').forEach((el) => {
      const value = normalized === 'zh'
        ? el.getAttribute('data-i18n-text-zh')
        : el.getAttribute('data-i18n-text-en');
      if (value) {
        el.textContent = value;
      }
    });

    document.querySelectorAll('[data-i18n-aria-label-zh]').forEach(el => {
      el.setAttribute('aria-label', el.getAttribute('data-i18n-aria-label-' + normalized));
    });

    document.dispatchEvent(new CustomEvent('langchange', { detail: { lang: normalized } }));
  }

  window.getCurrentLang = function () {
    return document.documentElement.getAttribute('data-lang') || defaultLang;
  };

  window.toggleLang = function () {
    const current = window.getCurrentLang();
    applyLang(current === 'zh' ? 'en' : 'zh');
  };

  let stored = defaultLang;
  try { stored = localStorage.getItem('lang') || defaultLang; } catch (error) {}
  applyLang(stored);
})();
