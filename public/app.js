const manualForm = document.getElementById("manualForm");
const batchForm = document.getElementById("batchForm");
const datasetInput = document.getElementById("dataset");
const manualModelSelect = document.getElementById("manualModel");
const batchModelSelect = document.getElementById("batchModel");
const batchSubmitButton = document.getElementById("batchSubmitButton");
const runBatchButton = document.getElementById("runBatchButton");
const statusBox = document.getElementById("statusBox");
const selectedModel = document.getElementById("selectedModel");
const predictionMode = document.getElementById("predictionMode");
const predictedRisk = document.getElementById("predictedRisk");
const predictionConfidence = document.getElementById("predictionConfidence");
const predictionCount = document.getElementById("predictionCount");
const accuracy = document.getElementById("accuracy");
const weightedF1 = document.getElementById("weightedF1");
const macroF1 = document.getElementById("macroF1");
const modelParametersBox = document.getElementById("modelParametersBox");
const modelParameters = document.getElementById("modelParameters");
const confusionMatrix = document.getElementById("confusionMatrix");
const classificationReport = document.getElementById("classificationReport");
const predictionTable = document.getElementById("predictionTable");
const downloadLink = document.getElementById("downloadLink");
const metricsNote = document.getElementById("metricsNote");
const warningsBox = document.getElementById("warningsBox");
const featureContributions = document.getElementById("featureContributions");
const manualFormFields = document.getElementById("manualFormFields");
const manualModeTab = document.getElementById("manualModeTab");
const batchModeTab = document.getElementById("batchModeTab");
const manualPanel = document.getElementById("manualPanel");
const batchPanel = document.getElementById("batchPanel");
const compatibilityPanel = document.getElementById("compatibilityPanel");
const compatibilitySummary = document.getElementById("compatibilitySummary");
const compatibilityReport = document.getElementById("compatibilityReport");
const featureMappingTableBody = document.getElementById("featureMappingTableBody");
const rawFieldMappingTableBody = document.getElementById("rawFieldMappingTableBody");
const optionalIdentifiersTableBody = document.getElementById("optionalIdentifiersTableBody");
const unsupportedColumnsList = document.getElementById("unsupportedColumnsList");
const compatibilityState = document.getElementById("compatibilityState");

const SECTOR_OPTIONS = [
  "Basic Industries",
  "Capital Goods",
  "Consumer Durables",
  "Consumer Non-Durables",
  "Consumer Services",
  "Energy",
  "Finance",
  "Health Care",
  "Miscellaneous",
  "Public Utilities",
  "Technology",
  "Transportation"
];

const MANUAL_FIELD_SECTIONS = [
  {
    title: "Company Info",
    fields: [
      { name: "company_name", label: "Company Name", type: "text", placeholder: "Whirlpool Corporation", optional: true, helpText: "Use the legal or trading name shown in your records." },
      { name: "symbol", label: "Ticker Symbol", type: "text", placeholder: "WHR", optional: true, helpText: "Use the stock code or internal issuer symbol if available." },
      { name: "sector", label: "Sector", type: "select", required: true, options: SECTOR_OPTIONS }
    ]
  },
  {
    title: "Liquidity",
    fields: [
      { name: "current_assets", label: "Current Assets", type: "number", step: "any", required: true, helpText: "Enter all assets expected to be converted to cash within one year." },
      { name: "current_liabilities", label: "Current Liabilities", type: "number", step: "any", required: true, helpText: "Enter all obligations due within one year." },
      { name: "cash_and_equivalents", label: "Cash and Cash Equivalents", type: "number", step: "any", required: true, helpText: "Use cash on hand plus near-cash instruments such as bank balances." },
      { name: "inventory", label: "Inventory", type: "number", step: "any", required: true, helpText: "Enter the value of stock or goods held for sale." },
      { name: "accounts_receivable", label: "Accounts Receivable", type: "number", step: "any", required: true, helpText: "Use outstanding customer invoices or trade receivables." }
    ]
  },
  {
    title: "Income Statement",
    fields: [
      { name: "revenue", label: "Revenue", type: "number", step: "any", required: true, helpText: "Use total sales or turnover for the reporting period." },
      { name: "gross_profit", label: "Gross Profit", type: "number", step: "any", required: true, helpText: "Enter revenue minus cost of goods sold." },
      { name: "operating_income", label: "Operating Income", type: "number", step: "any", required: true, helpText: "Use earnings before interest and taxes from operations." },
      { name: "ebit", label: "EBIT", type: "number", step: "any", required: true, helpText: "Enter earnings before interest and tax." },
      { name: "net_income", label: "Net Income", type: "number", step: "any", required: true, helpText: "Use profit after all expenses and tax." },
      { name: "pretax_income", label: "Pre-tax Income", type: "number", step: "any", required: true, helpText: "Enter earnings before tax expense." },
      { name: "tax_expense", label: "Tax Expense", type: "number", step: "any", required: true, helpText: "Use income tax charged for the period." }
    ]
  },
  {
    title: "Balance Sheet and Cash Flow",
    fields: [
      { name: "total_assets", label: "Total Assets", type: "number", step: "any", required: true, helpText: "Enter the total book value of assets." },
      { name: "net_fixed_assets", label: "Net Fixed Assets", type: "number", step: "any", required: true, helpText: "Use property, plant and equipment after depreciation." },
      { name: "total_debt", label: "Total Debt", type: "number", step: "any", required: true, helpText: "Enter all short-term and long-term debt." },
      { name: "shareholders_equity", label: "Shareholders' Equity", type: "number", step: "any", required: true, helpText: "Use total equity attributable to owners." },
      { name: "free_cash_flow", label: "Free Cash Flow", type: "number", step: "any", required: true, helpText: "Use operating cash flow minus capital expenditure." },
      { name: "operating_cash_flow", label: "Operating Cash Flow", type: "number", step: "any", required: true, helpText: "Enter cash generated from core operations." },
      { name: "shares_outstanding", label: "Shares Outstanding", type: "number", step: "any", required: true, helpText: "Use weighted average shares or total outstanding shares." },
      { name: "enterprise_value", label: "Enterprise Value", type: "number", step: "any", required: true, helpText: "Enter market value of equity plus debt minus cash." },
      { name: "ebitda", label: "EBITDA", type: "number", step: "any", required: true, helpText: "Use earnings before interest, tax, depreciation and amortisation." },
      { name: "accounts_payable", label: "Accounts Payable", type: "number", step: "any", required: true, helpText: "Enter outstanding supplier payables." }
    ]
  }
];

let currentPredictions = [];
let currentCompatibility = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showStatus(message, type) {
  statusBox.className = `status ${type}`;
  statusBox.textContent = message;
  statusBox.classList.remove("hidden");
}

function clearStatus() {
  statusBox.className = "status hidden";
  statusBox.textContent = "";
}

function clearResults() {
  selectedModel.textContent = "-";
  predictionMode.textContent = "-";
  predictedRisk.textContent = "-";
  predictionConfidence.textContent = "-";
  predictionCount.textContent = "-";
  accuracy.textContent = "-";
  weightedF1.textContent = "-";
  macroF1.textContent = "-";
  modelParametersBox.classList.add("hidden");
  modelParameters.innerHTML = "";
  metricsNote.textContent = "These metrics come from Dataset A's held-out test set, not the uploaded company file.";
  warningsBox.className = "warnings hidden";
  warningsBox.innerHTML = "";
  featureContributions.className = "contribution-box empty-state";
  featureContributions.textContent = "Run an analysis to see the SHAP explanation.";
  confusionMatrix.innerHTML = '<div class="empty-state">Run an analysis to see the matrix.</div>';
  classificationReport.textContent = "Run an analysis to see the report.";
  predictionTable.querySelector("tbody").innerHTML =
    '<tr><td colspan="5" class="empty-state">Run an analysis to see prediction rows.</td></tr>';
  downloadLink.href = "#";
  downloadLink.classList.add("disabled");
  currentPredictions = [];
  if (runBatchButton) {
    runBatchButton.disabled = true;
  }
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatConfidence(value) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  return Number(value).toFixed(4);
}

function renderConfusionMatrix(matrix, labels) {
  const header = ["<tr><th>Actual \\ Predicted</th>", ...labels.map((label) => `<th>${escapeHtml(label)}</th>`), "</tr>"].join("");
  const rows = matrix
    .map((row, index) => {
      const cells = row.map((value) => `<td>${value}</td>`).join("");
      return `<tr><th>${escapeHtml(labels[index])}</th>${cells}</tr>`;
    })
    .join("");

  confusionMatrix.innerHTML = `<table><thead>${header}</thead><tbody>${rows}</tbody></table>`;
}

function renderWarnings(warnings) {
  if (!warnings || !warnings.length) {
    warningsBox.className = "warnings hidden";
    warningsBox.innerHTML = "";
    return;
  }

  warningsBox.className = "warnings";
  warningsBox.innerHTML = `<strong>Warnings</strong><ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`;
}

function renderModelParameters(payload) {
  const parameters = payload.model_parameters || {};
  const entries = Object.entries(parameters);
  if (!entries.length || payload.model_name !== "xgboost") {
    modelParametersBox.classList.add("hidden");
    modelParameters.innerHTML = "";
    return;
  }

  const preferredOrder = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "random_state",
    "eval_metric"
  ];
  const orderedEntries = preferredOrder
    .filter((key) => key in parameters)
    .map((key) => [key, parameters[key]]);

  modelParameters.innerHTML = orderedEntries
    .map(([key, value]) => {
      const label = key.replaceAll("_", " ");
      return `<div class="model-parameter"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
    })
    .join("");

  modelParametersBox.classList.remove("hidden");
}

function setCompatibilityVisible(isVisible) {
  compatibilityPanel.classList.toggle("hidden", !isVisible);
}

function renderCompatibilitySummary(report) {
  const items = [
    { label: "Compatible", value: report.compatible ? "Yes" : "No" },
    { label: "Suggested mode", value: report.suggested_mode || "-" },
    { label: "Required features", value: report.required_features?.length ?? 0 },
    { label: "Missing features", value: report.missing_required_features?.length ?? 0 },
    { label: "Unsupported columns", value: report.unsupported_columns?.length ?? 0 }
  ];

  compatibilitySummary.innerHTML = items
    .map((item) => `<div><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`)
    .join("");
}

function renderCompatibilityReportText(report) {
  const parts = [];
  if (report.different_domain) {
    parts.push("The uploaded file does not appear to belong to the same credit-risk feature domain as the selected model.");
  } else if (report.compatible) {
    parts.push("The file is compatible with the selected model.");
  } else {
    parts.push("The file is missing required features for the selected model.");
  }
  if (report.missing_required_features?.length) {
    parts.push(`Missing required features: ${report.missing_required_features.join(", ")}.`);
  }
  compatibilityReport.textContent = parts.join(" ");
}

function buildSelectOptions(uploadedColumns, selectedValue) {
  const choices = ["<option value=\"\">Select a column</option>"];
  for (const column of uploadedColumns) {
    const selected = column === selectedValue ? " selected" : "";
    choices.push(`<option value="${escapeHtml(column)}"${selected}>${escapeHtml(column)}</option>`);
  }
  return choices.join("");
}

function renderCompatibilityTables(report) {
  const uploadedColumns = [
    ...(report.matched_columns || []),
    ...(report.unsupported_columns || [])
  ];

  featureMappingTableBody.innerHTML = (report.required_features || [])
    .map((row) => {
      const statusLabel = row.status === "calculated" ? "Will be calculated" : row.status === "matched" ? "Matched" : "Missing";
      const sourceLabel = row.source_column ? escapeHtml(row.source_column) : (row.source_columns || []).join(", ");
      const selectHtml = row.status === "missing"
        ? "<span class=\"mapping-missing\">No reliable mapping found</span>"
        : `<select data-feature-key="${escapeHtml(row.feature)}" ${row.status === "calculated" ? "disabled" : ""}>${buildSelectOptions(uploadedColumns, row.source_column || "")}</select>`;
      const formulaLabel = row.formula ? `${escapeHtml(row.formula)}${row.source_columns && row.source_columns.length ? ` using ${escapeHtml(row.source_columns.join(", "))}` : ""}` : sourceLabel || "-";
      return `<tr>
        <td>${escapeHtml(row.feature)}</td>
        <td><span class="status-pill ${row.status}">${statusLabel}</span></td>
        <td>${selectHtml}</td>
        <td>${formulaLabel || "-"}</td>
      </tr>`;
    })
    .join("");

  rawFieldMappingTableBody.innerHTML = (report.raw_source_fields || [])
    .map((row) => {
      const statusLabel = row.status === "matched" ? "Matched" : "Missing";
      const selectHtml = row.status === "missing"
        ? "<span class=\"mapping-missing\">Not found</span>"
        : `<select data-raw-key="${escapeHtml(row.raw_field)}">${buildSelectOptions(uploadedColumns, row.input_column || "")}</select>`;
      return `<tr>
        <td>${escapeHtml(row.raw_field.replaceAll("_", " "))}</td>
        <td><span class="status-pill ${row.status}">${statusLabel}</span></td>
        <td>${selectHtml}</td>
      </tr>`;
    })
    .join("");

  optionalIdentifiersTableBody.innerHTML = (report.optional_identifiers || [])
    .map((row) => {
      const statusLabel = row.status === "matched" ? "Matched" : "Missing";
      return `<tr>
        <td>${escapeHtml(row.matched_as || row.column_type || "Identifier")}</td>
        <td>${escapeHtml(row.input_column || "-")}</td>
        <td><span class="status-pill ${row.status}">${statusLabel}</span></td>
      </tr>`;
    })
    .join("");

  if (report.unsupported_columns?.length) {
    unsupportedColumnsList.className = "compatibility-list";
    unsupportedColumnsList.innerHTML = `<ul>${report.unsupported_columns.map((column) => `<li>${escapeHtml(column)}</li>`).join("")}</ul>`;
  } else {
    unsupportedColumnsList.className = "compatibility-list empty-state";
    unsupportedColumnsList.textContent = "No unsupported columns were found.";
  }
}

function renderCompatibility(report) {
  currentCompatibility = report;
  setCompatibilityVisible(true);
  renderCompatibilitySummary(report);
  renderCompatibilityReportText(report);
  renderCompatibilityTables(report);
  compatibilityState.textContent = report.compatible
    ? "Review the financial input mappings below, then run the batch prediction."
    : "This file cannot be predicted until the required mappings are resolved.";
  runBatchButton.disabled = !report.compatible;
}

function collectCompatibilityMapping() {
  const featureMappings = {};
  featureMappingTableBody.querySelectorAll("select[data-feature-key]").forEach((select) => {
    featureMappings[select.dataset.featureKey] = select.value;
  });

  const rawFieldMappings = {};
  rawFieldMappingTableBody.querySelectorAll("select[data-raw-key]").forEach((select) => {
    rawFieldMappings[select.dataset.rawKey] = select.value;
  });

  return {
    mode: currentCompatibility?.suggested_mode || "direct",
    feature_mappings: featureMappings,
    raw_field_mappings: rawFieldMappings
  };
}

function renderFeatureContributions(prediction) {
  if (!prediction || !prediction.top_contributions || !prediction.top_contributions.length) {
    featureContributions.className = "contribution-box empty-state";
    featureContributions.textContent = "No SHAP explanation was returned for this row.";
    return;
  }

  const companyLabel = prediction.company_name ? `${prediction.company_name}` : "This company";
  const contributions = prediction.top_contributions.map((item) => ({
    ...item,
    shapValue: Number(item.shap_value) || 0,
    magnitude: Math.abs(Number(item.shap_value) || 0)
  }));
  const maxMagnitude = Math.max(...contributions.map((item) => item.magnitude), 0.000001);
  const items = contributions
    .map((item) => {
      const directionText = item.shapValue < 0 ? "Pushes away" : "Pushes toward";
      const width = Math.max(8, (item.magnitude / maxMagnitude) * 100);
      return `<li class="shap-item">
        <div class="shap-item-head">
          <strong>${escapeHtml(item.feature)}</strong>
          <span class="shap-pill ${item.shapValue < 0 ? "negative" : "positive"}">${directionText}</span>
        </div>
        <div class="shap-bar" aria-hidden="true"><span style="width: ${width}%"></span></div>
        <div class="shap-item-meta">SHAP value: ${item.shapValue >= 0 ? "+" : ""}${item.shapValue.toFixed(4)}</div>
      </li>`;
    })
    .join("");

  featureContributions.className = "contribution-box";
  featureContributions.innerHTML = `
    <div class="shap-summary">
      <p class="shap-summary-title"><strong>${escapeHtml(companyLabel)}</strong> was predicted as <strong>${escapeHtml(prediction.predicted_rating_group)}</strong>.</p>
      <p class="shap-summary-copy">The list below shows the strongest features behind that result, ordered from biggest effect to smallest.</p>
    </div>
    <ul class="shap-list">${items}</ul>
  `;
}

function renderPredictionRows(rows) {
  const body = predictionTable.querySelector("tbody");
  currentPredictions = rows;

  if (!rows || !rows.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty-state">Run an analysis to see prediction rows.</td></tr>';
    renderFeatureContributions(null);
    return;
  }

  body.innerHTML = rows
    .map(
      (row, index) => `
        <tr data-row-index="${index}" tabindex="0">
          <td>${row.row_index}</td>
          <td>${escapeHtml(row.company_name || "-")}</td>
          <td>${escapeHtml(row.predicted_rating_group)}</td>
          <td>${formatConfidence(row.confidence_score)}</td>
          <td>${escapeHtml(row.top_contributions_text || "Unavailable")}</td>
        </tr>`
    )
    .join("");

  const tableRows = body.querySelectorAll("tr[data-row-index]");
  tableRows.forEach((rowElement) => {
    const index = Number(rowElement.dataset.rowIndex);
    const activate = () => {
      body.querySelectorAll("tr[data-row-index]").forEach((item) => item.classList.remove("selected-row"));
      rowElement.classList.add("selected-row");
      renderFeatureContributions(currentPredictions[index]);
    };
    rowElement.addEventListener("click", activate);
    rowElement.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });

  renderFeatureContributions(currentPredictions[0]);
  body.querySelector("tr[data-row-index]")?.classList.add("selected-row");
}

function populateManualForm() {
  manualFormFields.innerHTML = MANUAL_FIELD_SECTIONS.map((section) => {
    const fieldsHtml = section.fields
      .map((field) => {
        const required = field.required ? "required" : "";
        const optionalLabel = field.optional ? '<span class="field-optional">Optional</span>' : "";
        const helpText = field.helpText ? `<small class="field-help">${escapeHtml(field.helpText)}</small>` : "";
        if (field.type === "select") {
          const options = field.options
            .map((option) => `<option value="${escapeHtml(option)}">${escapeHtml(option)}</option>`)
            .join("");
          return `
            <label>
              <span>${escapeHtml(field.label)} ${optionalLabel}</span>
              <select name="${escapeHtml(field.name)}" ${required}>
                <option value="">Select a sector</option>
                ${options}
              </select>
              ${helpText}
            </label>`;
        }

        return `
          <label>
            <span>${escapeHtml(field.label)} ${optionalLabel}</span>
            <input
              name="${escapeHtml(field.name)}"
              type="${field.type}"
              ${field.step ? `step="${field.step}"` : ""}
              ${field.placeholder ? `placeholder="${escapeHtml(field.placeholder)}"` : ""}
              ${required}
            />
            ${helpText}
          </label>`;
      })
      .join("");

    return `
      <fieldset class="manual-section">
        <legend>${escapeHtml(section.title)}</legend>
        <div class="manual-grid">${fieldsHtml}</div>
      </fieldset>`;
  }).join("");
}

function setMode(mode) {
  const manualActive = mode === "manual";
  manualModeTab.classList.toggle("active", manualActive);
  batchModeTab.classList.toggle("active", !manualActive);
  manualModeTab.setAttribute("aria-selected", String(manualActive));
  batchModeTab.setAttribute("aria-selected", String(!manualActive));
  manualPanel.classList.toggle("hidden", !manualActive);
  batchPanel.classList.toggle("hidden", manualActive);
  clearStatus();
}

function setLoadingState(isLoading, submitButton) {
  if (!submitButton.dataset.label) {
    submitButton.dataset.label = submitButton.textContent;
  }
  manualForm.querySelectorAll("button, input, select").forEach((element) => {
    element.disabled = isLoading && element !== submitButton;
  });
  batchForm.querySelectorAll("button, input, select").forEach((element) => {
    element.disabled = isLoading && element !== submitButton;
  });
  if (runBatchButton) {
    runBatchButton.disabled = isLoading || !(currentCompatibility && currentCompatibility.compatible);
  }
  submitButton.textContent = isLoading ? "Running..." : submitButton.dataset.label;
}

function collectManualPayload() {
  const formData = new FormData(manualForm);
  const payload = Object.fromEntries(formData.entries());
  for (const field of MANUAL_FIELD_SECTIONS.flatMap((section) => section.fields)) {
    if (field.type === "number") {
      const value = payload[field.name];
      payload[field.name] = value === "" ? null : value;
    }
  }
  return payload;
}

async function submitManualAssessment(event) {
  event.preventDefault();
  clearResults();
  clearStatus();

  const submitButton = manualForm.querySelector("button[type='submit']");
  setLoadingState(true, submitButton);
  try {
    const payload = collectManualPayload();
    payload.model = manualModelSelect.value;

    showStatus("Running manual assessment. Please wait...", "success");

    const response = await fetch("/api/analyze-manual", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const responsePayload = await response.json();
    if (!response.ok) {
      const detailText = responsePayload.details ? ` ${typeof responsePayload.details === "string" ? responsePayload.details : JSON.stringify(responsePayload.details)}` : "";
      throw new Error(`${responsePayload.error || "Manual assessment failed."}${detailText}`);
    }

    renderResponse(responsePayload);
    showStatus("Manual assessment completed.", "success");
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    setLoadingState(false, submitButton);
  }
}

async function checkBatchCompatibility(event) {
  event.preventDefault();
  clearStatus();

  if (!datasetInput.files.length) {
    showStatus("Please upload a dataset file first.", "error");
    return;
  }

  clearResults();
  setCompatibilityVisible(false);
  const submitButton = batchSubmitButton;
  setLoadingState(true, submitButton);
  try {
    const formData = new FormData();
    formData.append("dataset", datasetInput.files[0]);
    formData.append("model", batchModelSelect.value);

    showStatus("Checking dataset compatibility. Please wait...", "success");

    const response = await fetch("/api/compatibility", {
      method: "POST",
      body: formData
    });

    const responsePayload = await response.json();
    if (!response.ok) {
      const detailText = responsePayload.details ? ` ${typeof responsePayload.details === "string" ? responsePayload.details : JSON.stringify(responsePayload.details)}` : "";
      throw new Error(`${responsePayload.error || "Compatibility check failed."}${detailText}`);
    }

    renderCompatibility(responsePayload);
    showStatus("Compatibility check completed. Review the mappings, then run prediction.", "success");
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    setLoadingState(false, submitButton);
  }
}

async function runBatchPrediction() {
  if (!datasetInput.files.length) {
    showStatus("Please upload a dataset file first.", "error");
    return;
  }

  if (!currentCompatibility) {
    showStatus("Check dataset compatibility before running prediction.", "error");
    return;
  }

  const submitButton = runBatchButton;
  setLoadingState(true, submitButton);
  try {
    const formData = new FormData();
    formData.append("dataset", datasetInput.files[0]);
    formData.append("model", batchModelSelect.value);
    formData.append("mapping", JSON.stringify(collectCompatibilityMapping()));

    showStatus("Running batch prediction. Please wait...", "success");

    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    const responsePayload = await response.json();
    if (!response.ok) {
      const detailText = responsePayload.details ? ` ${typeof responsePayload.details === "string" ? responsePayload.details : JSON.stringify(responsePayload.details)}` : "";
      throw new Error(`${responsePayload.error || "Batch prediction failed."}${detailText}`);
    }

    renderResponse(responsePayload);
    showStatus(`Batch prediction completed for ${responsePayload.prediction_count} rows.`, "success");
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    setLoadingState(false, submitButton);
  }
}

function renderResponse(payload) {
  selectedModel.textContent = payload.model_display_name || payload.model_name || "-";
  predictionMode.textContent = payload.prediction_mode || "-";
  predictionCount.textContent = payload.prediction_count ?? "-";
  accuracy.textContent = formatPercent(payload.metrics?.baseline_test_accuracy);
  weightedF1.textContent = formatPercent(payload.metrics?.baseline_test_weighted_f1);
  macroF1.textContent = formatPercent(payload.metrics?.baseline_test_macro_f1);
  renderModelParameters(payload);
  metricsNote.textContent = payload.metrics_note || metricsNote.textContent;
  renderConfusionMatrix(payload.confusion_matrix, payload.class_labels || []);
  classificationReport.textContent = payload.classification_report_text || "Unavailable";
  renderPredictionRows(payload.predictions || []);
  renderWarnings(payload.warnings || []);

  const primaryPrediction = payload.predicted_risk_category || payload.predictions?.[0]?.predicted_rating_group || "-";
  const confidence = payload.confidence_score ?? payload.predictions?.[0]?.confidence_score;
  predictedRisk.textContent = primaryPrediction;
  predictionConfidence.textContent = formatConfidence(confidence);

  if (payload.top_feature_contributions && payload.top_feature_contributions.length) {
    renderFeatureContributions({
      predicted_rating_group: primaryPrediction,
      company_name: payload.predictions?.[0]?.company_name || payload.manual_metadata?.company_name || "",
      top_contributions: payload.top_feature_contributions
    });
  }

  if (payload.output_csv_url) {
    downloadLink.href = payload.output_csv_url;
    downloadLink.classList.remove("disabled");
    downloadLink.textContent = "Download CSV";
  }
}

manualModeTab.addEventListener("click", () => setMode("manual"));
batchModeTab.addEventListener("click", () => setMode("batch"));
manualForm.addEventListener("submit", submitManualAssessment);
batchForm.addEventListener("submit", checkBatchCompatibility);
runBatchButton.addEventListener("click", runBatchPrediction);
datasetInput.addEventListener("change", () => {
  currentCompatibility = null;
  setCompatibilityVisible(false);
  runBatchButton.disabled = true;
});
batchModelSelect.addEventListener("change", () => {
  currentCompatibility = null;
  setCompatibilityVisible(false);
  runBatchButton.disabled = true;
});

populateManualForm();
setMode("manual");
clearResults();
