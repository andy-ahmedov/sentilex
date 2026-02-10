document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("[data-upload-form]");
  if (!form) {
    return;
  }

  const dropzone = form.querySelector("[data-dropzone]");
  const fileInput = form.querySelector("[data-file-input]");
  const fileName = form.querySelector("[data-file-name]");
  const pickButton = form.querySelector("[data-pick-file]");
  const submitButton = form.querySelector("[data-submit-btn]");
  const submitSpinner = form.querySelector("[data-submit-spinner]");
  const submitLabel = form.querySelector("[data-submit-label]");
  const chunkRange = form.querySelector("[data-chunk-range]");
  const chunkInput = form.querySelector("[data-chunk-input]");

  const updateFileName = (file) => {
    fileName.textContent = file ? file.name : "Файл не выбран";
  };

  const attachFile = (file) => {
    if (!file) {
      return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    updateFileName(file);
  };

  pickButton.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    updateFileName(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const droppedFile = event.dataTransfer.files[0];
    attachFile(droppedFile);
  });

  dropzone.addEventListener("click", () => fileInput.click());

  const clampChunkValue = (value) => {
    const parsed = Number(value);
    if (Number.isNaN(parsed)) {
      return Number(chunkInput.min || 10);
    }
    const min = Number(chunkInput.min || 10);
    const max = Number(chunkInput.max || 500);
    return Math.max(min, Math.min(max, parsed));
  };

  if (chunkRange && chunkInput) {
    chunkRange.addEventListener("input", () => {
      chunkInput.value = chunkRange.value;
    });

    chunkInput.addEventListener("input", () => {
      chunkRange.value = clampChunkValue(chunkInput.value);
    });

    chunkInput.addEventListener("blur", () => {
      const normalized = clampChunkValue(chunkInput.value);
      chunkInput.value = normalized;
      chunkRange.value = normalized;
    });
  }

  form.addEventListener("submit", () => {
    submitButton.disabled = true;
    submitSpinner.classList.remove("d-none");
    submitLabel.textContent = "Анализируем…";
  });
});
