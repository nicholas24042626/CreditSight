const form = document.getElementById("analysisForm");
const datasetInput = document.getElementById("dataset");
const modelSelect = document.getElementById("model");
const statusBox = document.getElementById("statusBox");
const selectedModel = document.getElementById("selectedModel");
const accuracy = document.getElementById("accuracy");
const weightedF1 = document.getElementById("weightedF1");
const macroF1 = document.getElementById("macroF1");
const confusionMatrix = document.getElementById("confusionMatrix");
const classificationReport = document.getElementById("classificationReport");
const predictionTable = document.getElementById("predictionTable");
const downloadLink = document.getElementById("downloadLink");
const metricsNote = document.getElementById("metricsNote");

// These small helpers keep the page updates easy to read.
function showStatus(message, type) {
  statusBox.className = `status ${type}`;
  statusBox.textContent = message;
  statusBox.classList.remove("hidden");
}

function clearResults() {
  selectedModel.textContent = "-";
  accuracy.textContent = "-";
  weightedF1.textContent = "-";
  macroF1.textContent = "-";
  metricsNote.textContent = "These metrics come from Dataset A's held-out test set, not the uploaded company file.";
  confusionMatrix.innerHTML = '<div class="empty-state">Run an analysis to see the matrix.</div>';
  classificationReport.textContent = "Run an analysis to see the report.";
  predictionTable.querySelector("tbody").innerHTML =
    '<tr><td colspan="3" class="empty-state">Run an analysis to see row-level predictions.</td></tr>';
  downloadLink.href = "#";
  downloadLink.classList.add("disabled");
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function renderConfusionMatrix(matrix, labels) {
  const header = ["<tr><th>Actual \\ Predicted</th>", ...labels.map((label) => `<th>${label}</th>`), "</tr>"].join("");
  const rows = matrix
    .map((row, index) => {
      const cells = row.map((value) => `<td>${value}</td>`).join("");
      return `<tr><th>${labels[index]}</th>${cells}</tr>`;
    })
    .join("");

  confusionMatrix.innerHTML = `<table><thead>${header}</thead><tbody>${rows}</tbody></table>`;
}

function renderPredictionRows(rows) {
  const body = predictionTable.querySelector("tbody");
  body.innerHTML = rows
    .map((row) => `<tr><td>${row.row_index}</td><td>${row.predicted_rating_group}</td><td>${row.confidence_score === null ? "N/A" : Number(row.confidence_score).toFixed(4)}</td></tr>`)
    .join("");
}

form.addEventListener("submit", async (event) => {
  // When the user clicks run, send the file to the backend.
  event.preventDefault();
  clearResults();

  if (!datasetInput.files.length) {
    showStatus("Please upload a dataset file first.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("dataset", datasetInput.files[0]);
  formData.append("model", modelSelect.value);

  showStatus("Running analysis. Please wait...", "success");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    const payload = await response.json();

    if (!response.ok) {
      const detailText = payload.details ? ` ${typeof payload.details === "string" ? payload.details : JSON.stringify(payload.details)}` : "";
      throw new Error(`${payload.error || "Analysis failed."}${detailText}`);
    }

    selectedModel.textContent = payload.model_display_name || payload.model_name || modelSelect.options[modelSelect.selectedIndex].text;
    accuracy.textContent = formatPercent(payload.metrics.baseline_test_accuracy);
    weightedF1.textContent = formatPercent(payload.metrics.baseline_test_weighted_f1);
    macroF1.textContent = formatPercent(payload.metrics.baseline_test_macro_f1);
    metricsNote.textContent = payload.metrics_note || metricsNote.textContent;
    renderConfusionMatrix(payload.confusion_matrix, payload.class_labels);
    classificationReport.textContent = payload.classification_report_text;
    renderPredictionRows(payload.predictions);

    if (payload.output_csv_url) {
      downloadLink.href = payload.output_csv_url;
      downloadLink.classList.remove("disabled");
      downloadLink.textContent = "Download CSV";
    }

    showStatus(`Analysis completed with ${payload.prediction_count} predicted rows.`, "success");
  } catch (error) {
    showStatus(error.message, "error");
  }
});
