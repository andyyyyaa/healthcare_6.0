(() => {
  const zh = () => document.documentElement.dataset.lang !== 'en';
  const toggle = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebar-backdrop');
  const main = document.querySelector('.main-content');
  const smallScreen = window.matchMedia('(max-width: 991px)');
  function setMenu(open) {
    sidebar.inert = smallScreen.matches && !open;
    sidebar.classList.toggle('show', open);
    toggle.setAttribute('aria-expanded', String(open));
    backdrop.hidden = !open;
    document.body.style.overflow = open ? 'hidden' : '';
    main.inert = open;
    if (open) sidebar.querySelector('a').focus();
    else toggle.focus({preventScroll: true});
  }
  toggle?.addEventListener('click', () => setMenu(!sidebar.classList.contains('show')));
  backdrop?.addEventListener('click', () => setMenu(false));
  document.addEventListener('keydown', event => {
    if (!sidebar.classList.contains('show')) return;
    if (event.key === 'Escape') setMenu(false);
    if (event.key === 'Tab') {
      const links = [...sidebar.querySelectorAll('a')];
      if (event.shiftKey && document.activeElement === links[0]) { event.preventDefault(); links.at(-1).focus(); }
      else if (!event.shiftKey && document.activeElement === links.at(-1)) { event.preventDefault(); links[0].focus(); }
    }
  });
  function updateSidebar() {
    if (!smallScreen.matches && sidebar.classList.contains('show')) setMenu(false);
    sidebar.inert = smallScreen.matches && !sidebar.classList.contains('show');
  }
  smallScreen.addEventListener('change', updateSidebar);
  updateSidebar();
  function updateDate() {
    const date = document.getElementById('workspace-date');
    if (date) { date.dateTime = new Date().toISOString().slice(0,10); date.textContent = new Intl.DateTimeFormat(zh() ? 'zh-CN' : 'en-US', {year:'numeric',month:'long',day:'numeric',weekday:'short'}).format(new Date()); }
  }
  updateDate();
  document.addEventListener('langchange', updateDate);

  const form = document.getElementById('xray-form');
  if (!form) return;
  const input = document.getElementById('fileInput');
  const zone = document.getElementById('drop-zone');
  const preview = document.getElementById('upload-preview');
  const image = document.getElementById('preview-image');
  const error = document.getElementById('upload-error');
  const submit = document.getElementById('analyze-button');
  const loading = document.getElementById('loading');
  let objectUrl;
  let errorType = '';
  const errors = {type:['请选择 JPG 或 PNG 格式的影像。','Please choose a JPG or PNG image.'],size:['影像大小不能超过 15 MB。','The image must be 15 MB or smaller.'],image:['无法读取这张影像，请重新选择。','This image could not be read. Please choose another.']};
  function showError(type) { errorType=type; error.hidden=!type; error.textContent=type ? errors[type][zh()?0:1] : ''; }
  function reset() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl=undefined; image.removeAttribute('src'); input.value=''; preview.hidden=true; zone.hidden=false; showError('');
  }
  function selected() {
    const file=input.files[0];
    if (!file) return;
    if (!['image/jpeg','image/png'].includes(file.type)) { reset(); showError('type'); return; }
    if (file.size>15*1024*1024) { reset(); showError('size'); return; }
    showError('');
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl=URL.createObjectURL(file);
    image.onload=() => { preview.hidden=false; zone.hidden=true; };
    image.onerror=() => { reset(); showError('image'); };
    image.src=objectUrl;
    document.getElementById('file-name').textContent=file.name;
    document.getElementById('file-size').textContent=(file.size/1024/1024).toFixed(2)+' MB';
  }
  input.addEventListener('change',selected);
  document.getElementById('remove-file').addEventListener('click',() => { reset(); input.focus(); });
  ['dragenter','dragover'].forEach(name => zone.addEventListener(name,e => {e.preventDefault();zone.classList.add('drag-over');}));
  ['dragleave','drop'].forEach(name => zone.addEventListener(name,e => {e.preventDefault();zone.classList.remove('drag-over');}));
  zone.addEventListener('drop',e => { const file=e.dataTransfer.files[0]; if(file){ const transfer=new DataTransfer();transfer.items.add(file);input.files=transfer.files;selected();} });
  form.addEventListener('submit',() => {submit.disabled=true;loading.hidden=false;form.setAttribute('aria-busy','true');});
  window.addEventListener('pageshow',() => {submit.disabled=false;loading.hidden=true;form.removeAttribute('aria-busy');});
  document.addEventListener('langchange',() => showError(errorType));
})();
