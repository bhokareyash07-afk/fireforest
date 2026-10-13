(() => {
  const form = document.querySelector('#analysis-form');
  if (!form) return;
  const maxBytes = 10 * 1024 * 1024;
  const accepted = { before: false, after: false };
  const pending = { before: false, after: false };
  const files = { before: null, after: null };
  const analyze = document.querySelector('#analyze-button');
  const label = analyze.querySelector('.button-label');
  const dots = analyze.querySelector('.loading-dots');

  const refreshButton = () => {
    analyze.disabled = !(accepted.before && accepted.after) || pending.before || pending.after;
  };
  const setStatus = (role, message, kind) => {
    const card = document.querySelector(`.upload-card[data-role="${role}"]`);
    const status = card.querySelector('.validation-status');
    status.className = `validation-status ${kind || ''}`;
    status.textContent = message;
  };
  const clearCard = (role) => {
    accepted[role] = false; pending[role] = false; files[role] = null;
    const card = document.querySelector(`.upload-card[data-role="${role}"]`);
    const input = card.querySelector('input'); const preview = card.querySelector('.preview-wrap');
    input.value = ''; preview.hidden = true; preview.querySelector('img').removeAttribute('src');
    setStatus(role, '', ''); refreshButton();
  };
  const validate = async (role, file) => {
    const card = document.querySelector(`.upload-card[data-role="${role}"]`);
    const input = card.querySelector('input');
    if (!file) return clearCard(role);
    accepted[role] = false; pending[role] = true; files[role] = file;
    if (file.size > maxBytes) {
      pending[role] = false; setStatus(role, 'This image is larger than the 10 MB upload limit.', 'invalid'); refreshButton(); return;
    }
    const preview = card.querySelector('.preview-wrap'); const img = preview.querySelector('img');
    img.src = URL.createObjectURL(file); preview.hidden = false;
    setStatus(role, 'Checking forest content…', 'loading'); refreshButton();
    const payload = new FormData(); payload.append('image', file, file.name);
    try {
      const response = await fetch('/api/validate-image', { method: 'POST', body: payload });
      const data = await response.json();
      if (input.files[0] !== file) return;
      pending[role] = false; accepted[role] = Boolean(data.ok);
      if (data.ok) setStatus(role, '✓ ' + data.message, 'accepted');
      else if (data.status === 'invalid') setStatus(role, 'Invalid Images: Forest Images Required. ' + data.message, 'invalid');
      else setStatus(role, data.message || 'This image could not be confidently validated. Please try a clearer forest scene.', data.status || 'error');
    } catch (_) {
      pending[role] = false; accepted[role] = false;
      setStatus(role, 'Could not validate this image. Check your connection and try again.', 'error');
    }
    refreshButton();
  };
  document.querySelectorAll('.upload-card input[type="file"]').forEach(input => {
    input.addEventListener('change', () => validate(input.dataset.role, input.files[0]));
  });
  document.querySelectorAll('.remove-file').forEach(button => {
    button.addEventListener('click', () => clearCard(button.closest('.upload-card').dataset.role));
  });
  form.addEventListener('submit', event => {
    if (!(accepted.before && accepted.after) || pending.before || pending.after) {
      event.preventDefault();
      setStatus('before', accepted.before ? '' : 'Please select and validate a suitable forest image.', accepted.before ? '' : 'invalid');
      setStatus('after', accepted.after ? '' : 'Please select and validate a suitable forest image.', accepted.after ? '' : 'invalid');
      return;
    }
    analyze.disabled = true; label.hidden = true; dots.hidden = false;
  });
})();
