/**
 * main.js — Medical Diagnose Frontend
 * Handles: chip selection, drag-and-drop upload, loading overlay
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── Element refs ─────────────────────────────────────────
  const textarea      = document.getElementById('symptoms');
  const chipsGrid     = document.getElementById('chipsGrid');
  const dropZone      = document.getElementById('dropZone');
  const imageInput    = document.getElementById('imageInput');
  const dropIdle      = document.getElementById('dropIdle');
  const dropPreview   = document.getElementById('dropPreview');
  const previewImg    = document.getElementById('previewImg');
  const previewName   = document.getElementById('previewName');
  const removeBtn     = document.getElementById('removeBtn');
  const form          = document.getElementById('diagnoseForm');
  const submitBtn     = document.getElementById('submitBtn');
  const loadingOverlay= document.getElementById('loadingOverlay');

  // ── Chip click → append to textarea ──────────────────────
  if (chipsGrid && textarea) {
    chipsGrid.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip');
      if (!chip) return;

      const symptom = chip.dataset.symptom;
      chip.classList.toggle('active');

      const current = textarea.value.trim();
      if (chip.classList.contains('active')) {
        // Add symptom
        textarea.value = current
          ? current.endsWith(',') || current.endsWith('.')
            ? current + ' ' + symptom.toLowerCase()
            : current + ', ' + symptom.toLowerCase()
          : symptom.toLowerCase();
      } else {
        // Remove symptom — strip it from textarea
        const pattern = new RegExp(
          '(,\\s*)?\\b' + escapeRegex(symptom.toLowerCase()) + '\\b(,\\s*)?',
          'gi'
        );
        textarea.value = textarea.value
          .replace(pattern, (match, before, after) => {
            if (before && after) return ', ';
            return '';
          })
          .replace(/^,\s*/, '')
          .replace(/,\s*$/, '')
          .trim();
      }
      textarea.dispatchEvent(new Event('input'));
    });
  }

  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // Sync chip active state if textarea is manually edited
  if (textarea && chipsGrid) {
    textarea.addEventListener('input', () => {
      const val = textarea.value.toLowerCase();
      chipsGrid.querySelectorAll('.chip').forEach(chip => {
        const sym = chip.dataset.symptom.toLowerCase();
        if (val.includes(sym)) {
          chip.classList.add('active');
        } else {
          chip.classList.remove('active');
        }
      });
    });
  }

  // ── Image upload — drag and drop ─────────────────────────
  if (dropZone && imageInput) {

    // Click on browse text opens file picker
    const browseTrigger = document.getElementById('dropBrowse');
    if (browseTrigger) {
      browseTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.click();
      });
    }

    // Click anywhere on drop zone opens file picker
    dropZone.addEventListener('click', (e) => {
      if (e.target === removeBtn) return;
      imageInput.click();
    });

    // Drag events
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    // File input change
    imageInput.addEventListener('change', () => {
      if (imageInput.files[0]) handleFile(imageInput.files[0]);
    });

    // Remove button
    if (removeBtn) {
      removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage();
      });
    }
  }

  function handleFile(file) {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp'];
    if (!allowed.includes(file.type)) {
      showFileError('Unsupported file type. Please use JPG, PNG, BMP, or WEBP.');
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showFileError('File too large. Maximum size is 16 MB.');
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg)  previewImg.src = e.target.result;
      if (previewName) previewName.textContent = file.name + ' (' + formatBytes(file.size) + ')';
      if (dropIdle)    dropIdle.style.display = 'none';
      if (dropPreview) dropPreview.style.display = 'flex';
      dropZone.style.borderStyle = 'solid';
      dropZone.style.borderColor = 'var(--accent)';
    };
    reader.readAsDataURL(file);

    // Assign to hidden file input via DataTransfer
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      imageInput.files = dt.files;
    } catch (_) {
      // Some older browsers don't support DataTransfer assignment — file still set via input
    }
  }

  function clearImage() {
    imageInput.value = '';
    if (previewImg)  previewImg.src = '';
    if (previewName) previewName.textContent = '';
    if (dropIdle)    dropIdle.style.display = '';
    if (dropPreview) dropPreview.style.display = 'none';
    dropZone.style.borderStyle = 'dashed';
    dropZone.style.borderColor = '';
  }

  function showFileError(msg) {
    const existing = document.querySelector('.file-error');
    if (existing) existing.remove();
    const el = document.createElement('div');
    el.className = 'alert-error file-error';
    el.style.marginTop = '10px';
    el.innerHTML = '<span>⚠</span> ' + msg;
    dropZone.after(el);
    setTimeout(() => el.remove(), 4000);
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  // ── Loading overlay on form submit ────────────────────────
  if (form && loadingOverlay) {
    form.addEventListener('submit', (e) => {
      const symptoms = textarea ? textarea.value.trim() : '';
      if (!symptoms) return; // Let browser validation handle

      // Show overlay
      loadingOverlay.classList.add('visible');
      if (submitBtn) submitBtn.disabled = true;

      // Animate pipeline steps
      const steps = ['ls1','ls2','ls3','ls4','ls5'];
      let current = 0;

      function advanceStep() {
        // Mark previous as done
        if (current > 0) {
          const prev = document.getElementById(steps[current - 1]);
          if (prev) {
            prev.classList.remove('active');
            prev.classList.add('done');
          }
        }
        if (current < steps.length) {
          const el = document.getElementById(steps[current]);
          if (el) el.classList.add('active');
          current++;
          if (current < steps.length) {
            setTimeout(advanceStep, 600);
          }
        }
      }

      setTimeout(advanceStep, 300);
    });
  }

  // ── Auto-resize textarea ──────────────────────────────────
  if (textarea) {
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 300) + 'px';
    });
  }

  // ── Animate result bars on results page ──────────────────
  const barFills = document.querySelectorAll('.diag-bar-fill[data-width]');
  if (barFills.length > 0) {
    barFills.forEach((bar, i) => {
      bar.style.width = '0%';
      setTimeout(() => {
        bar.style.width = bar.dataset.width + '%';
      }, 200 + i * 150);
    });
  }

  // Animate primary bar
  const primaryFill = document.querySelector('.primary-bar-fill');
  if (primaryFill) {
    const target = primaryFill.style.width;
    primaryFill.style.width = '0%';
    setTimeout(() => {
      primaryFill.style.width = target;
    }, 150);
  }

});
