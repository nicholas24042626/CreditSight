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
const overallRiskSummary = document.getElementById("overallRiskSummary");
const overallRiskSearch = document.getElementById("overallRiskSearch");
const confusionMatrix = document.getElementById("confusionMatrix");
const classificationReport = document.getElementById("classificationReport");
const predictionTable = document.getElementById("predictionTable");
const downloadLink = document.getElementById("downloadLink");
const metricsNote = document.getElementById("metricsNote");
const warningsBox = document.getElementById("warningsBox");
const featureContributions = document.getElementById("featureContributions");
const shapCompanySearch = document.getElementById("shapCompanySearch");
const shapCompanyApply = document.getElementById("shapCompanyApply");
const shapCompanyOptions = document.getElementById("shapCompanyOptions");
const shapTopTab = document.getElementById("shapTopTab");
const shapBottomTab = document.getElementById("shapBottomTab");
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
let currentOverallRiskAggregates = [];
let currentOverallRiskQuery = "";
let currentShapMode = "top";
let currentShapPredictionIndex = 0;
const CREDIT_RISK_ORDER = ["Investment-High", "Investment-Low", "Speculative", "Distressed"];

function getOverallCategoryClass(category) {
  if (category === "Distressed") {
    return "overall-category distressed";
  }
  if (category === "Speculative") {
    return "overall-category speculative";
  }
  return "overall-category investment";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeCompanyName(value) {
  return String(value || "").trim().toLowerCase();
}

function getUniqueCompanyNames(rows) {
  const uniqueNames = [];
  const seen = new Set();

  (rows || []).forEach((row) => {
    const companyName = String(row.company_name || "").trim();
    if (!companyName) {
      return;
    }

    const normalized = normalizeCompanyName(companyName);
    if (seen.has(normalized)) {
      return;
    }

    seen.add(normalized);
    uniqueNames.push(companyName);
  });

  return uniqueNames;
}

function populateShapCompanyOptions(rows) {
  if (!shapCompanyOptions) {
    return;
  }

  const companyNames = getUniqueCompanyNames(rows);
  shapCompanyOptions.innerHTML = companyNames
    .map((companyName) => `<option value="${escapeHtml(companyName)}"></option>`)
    .join("");

  if (shapCompanyApply) {
    shapCompanyApply.disabled = companyNames.length === 0;
  }
}

function getShapContributions(prediction) {
  if (!prediction) {
    return [];
  }

  if (Array.isArray(prediction.shap_contributions)) {
    return prediction.shap_contributions;
  }

  if (Array.isArray(prediction.top_contributions)) {
    return prediction.top_contributions;
  }

  return [];
}

function getShapContributionSlice(contributions, mode) {
  if (!contributions.length) {
    return [];
  }

  if (mode === "bottom") {
    const negativeContributions = contributions
      .filter((item) => (Number(item.shap_value) || 0) < 0)
      .slice()
      .sort((a, b) => Math.abs(Number(b.shap_value) || 0) - Math.abs(Number(a.shap_value) || 0));

    if (negativeContributions.length < 5) {
      return [];
    }

    return negativeContributions.slice(0, 5);
  }

  return contributions.slice(0, 5);
}

function updateShapTabState() {
  if (shapTopTab) {
    shapTopTab.classList.toggle("active", currentShapMode === "top");
    shapTopTab.setAttribute("aria-selected", String(currentShapMode === "top"));
    shapTopTab.disabled = !currentPredictions.length;
  }

  if (shapBottomTab) {
    shapBottomTab.classList.toggle("active", currentShapMode === "bottom");
    shapBottomTab.setAttribute("aria-selected", String(currentShapMode === "bottom"));
    shapBottomTab.disabled = !currentPredictions.length;
  }
}

function setShapMode(mode) {
  if (mode !== "top" && mode !== "bottom") {
    return;
  }

  currentShapMode = mode;
  updateShapTabState();

  if (currentPredictions.length) {
    setSelectedPredictionRow(currentShapPredictionIndex, false);
  }
}

function findPredictionIndexForCompany(companyQuery) {
  const normalizedQuery = normalizeCompanyName(companyQuery);
  if (!normalizedQuery) {
    return -1;
  }

  const exactMatchIndex = currentPredictions.findIndex((row) => normalizeCompanyName(row.company_name) === normalizedQuery);
  if (exactMatchIndex !== -1) {
    return exactMatchIndex;
  }

  return currentPredictions.findIndex((row) => {
    const normalizedCompanyName = normalizeCompanyName(row.company_name);
    return normalizedCompanyName && (normalizedCompanyName.includes(normalizedQuery) || normalizedQuery.includes(normalizedCompanyName));
  });
}

function setSelectedPredictionRow(index, syncCompanySearch = false) {
  const body = predictionTable.querySelector("tbody");
  const tableRows = body.querySelectorAll("tr[data-row-index]");
  const selectedRow = tableRows[index];

  if (!selectedRow) {
    return false;
  }

  currentShapPredictionIndex = index;
  tableRows.forEach((item) => item.classList.remove("selected-row"));
  selectedRow.classList.add("selected-row");
  renderFeatureContributions(currentPredictions[index]);

  if (syncCompanySearch && shapCompanySearch) {
    shapCompanySearch.value = currentPredictions[index]?.company_name || "";
  }

  return true;
}

function showShapForSelectedCompany() {
  if (!currentPredictions.length) {
    featureContributions.className = "contribution-box empty-state";
    featureContributions.textContent = "Run an analysis to see the SHAP explanation.";
    return;
  }

  const query = shapCompanySearch ? shapCompanySearch.value.trim() : "";
  if (!query) {
    setSelectedPredictionRow(0, false);
    return;
  }

  const matchedIndex = findPredictionIndexForCompany(query);
  if (matchedIndex === -1) {
    featureContributions.className = "contribution-box empty-state";
    featureContributions.textContent = `No SHAP explanation was found for "${query}".`;
    return;
  }

  setSelectedPredictionRow(matchedIndex, true);
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
  currentOverallRiskAggregates = [];
  currentOverallRiskQuery = "";
  if (overallRiskSearch) {
    overallRiskSearch.value = "";
  }
  overallRiskSummary.className = "overall-risk-summary empty-state";
  overallRiskSummary.innerHTML = "Run an analysis to see company-level credit risk categories.";
  metricsNote.textContent = "Model metrics shown here are based on Dataset A's test set, not the uploaded file.";
  warningsBox.className = "warnings hidden";
  warningsBox.innerHTML = "";
  featureContributions.className = "contribution-box empty-state";
  featureContributions.textContent = "Run an analysis to see the SHAP explanation.";
  currentShapMode = "top";
  currentShapPredictionIndex = 0;
  if (shapCompanySearch) {
    shapCompanySearch.value = "";
  }
  populateShapCompanyOptions([]);
  updateShapTabState();
  confusionMatrix.innerHTML = '<div class="empty-state">Run an analysis to see the matrix.</div>';
  classificationReport.textContent = "Run an analysis to see the report.";
  predictionTable.querySelector("tbody").innerHTML =
    '<tr><td colspan="6" class="empty-state">Run an analysis to see prediction rows.</td></tr>';
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

function formatClassDistribution(probabilities) {
  if (!probabilities || typeof probabilities !== "object") {
    return "Unavailable";
  }

  return Object.entries(probabilities)
    .map(([label, probability]) => `${label}: ${formatPercent(probability)}`)
    .join(", ");
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

function getCompanyKey(row, fallbackIndex) {
  const companyName = String(row.company_name || "").trim();
  if (companyName) {
    return companyName;
  }
  return `Row ${fallbackIndex}`;
}

function aggregatePredictionsByCompany(rows) {
  const groups = new Map();
  (rows || []).forEach((row, index) => {
    const companyKey = getCompanyKey(row, index + 1);
    if (!groups.has(companyKey)) {
      groups.set(companyKey, {
        company_name: companyKey,
        total_records: 0,
        confidence_sum: 0,
        category_counts: Object.fromEntries(CREDIT_RISK_ORDER.map((category) => [category, 0]))
      });
    }

    const group = groups.get(companyKey);
    const category = row.predicted_rating_group || "Unknown";
    group.total_records += 1;
    group.confidence_sum += Number(row.confidence_score || 0);
    if (!(category in group.category_counts)) {
      group.category_counts[category] = 0;
    }
    group.category_counts[category] += 1;
  });

  return Array.from(groups.values()).map((group) => {
    let overall_category = CREDIT_RISK_ORDER[0];
    let best_count = -1;
    for (const category of CREDIT_RISK_ORDER) {
      const count = group.category_counts[category] || 0;
      if (count > best_count) {
        best_count = count;
        overall_category = category;
      }
    }

    return {
      ...group,
      overall_category,
      average_confidence: group.total_records > 0 ? group.confidence_sum / group.total_records : 0
    };
  });
}

function renderOverallRiskSummary(rows) {
  currentOverallRiskAggregates = aggregatePredictionsByCompany(rows);
  renderOverallRiskSummaryTable();
}

function renderOverallRiskSummaryTable() {
  const query = currentOverallRiskQuery.trim().toLowerCase();
  const aggregates = query
    ? currentOverallRiskAggregates.filter((group) => group.company_name.toLowerCase().includes(query))
    : currentOverallRiskAggregates;

  if (!currentOverallRiskAggregates.length) {
    overallRiskSummary.className = "overall-risk-summary empty-state";
    overallRiskSummary.innerHTML = "Run an analysis to see company-level credit risk categories.";
    return;
  }

  if (!aggregates.length) {
    overallRiskSummary.className = "overall-risk-summary empty-state";
    overallRiskSummary.innerHTML = "No companies match the current search.";
    return;
  }

  const header = `
    <tr>
      <th>Company</th>
      <th>Total Records</th>
      <th>Investment-High</th>
      <th>Investment-Low</th>
      <th>Speculative</th>
      <th>Distressed</th>
      <th>Average Confidence</th>
      <th>Overall Category</th>
    </tr>`;

  const rowsHtml = aggregates
    .map((group) => `
      <tr>
        <td>${escapeHtml(group.company_name)}</td>
        <td>${group.total_records}</td>
        <td>${group.category_counts["Investment-High"] || 0}</td>
        <td>${group.category_counts["Investment-Low"] || 0}</td>
        <td>${group.category_counts["Speculative"] || 0}</td>
        <td>${group.category_counts["Distressed"] || 0}</td>
        <td>${formatConfidence(group.average_confidence)}</td>
        <td><span class="${getOverallCategoryClass(group.overall_category)}">${escapeHtml(group.overall_category)}</span></td>
      </tr>`)
    .join("");

  overallRiskSummary.className = "overall-risk-summary";
  overallRiskSummary.innerHTML = `
    <div class="overall-risk-meta">
      <strong>${aggregates.length}</strong>
      <span>company groups found</span>
    </div>
    <div class="table-wrap">
      <table class="overall-risk-table">
        <thead>${header}</thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
  `;
}

function formatModelParameters(params) {
  const entries = Object.entries(params || {});
  if (!entries.length) {
    return "-";
  }

  const priority = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"];
  const ordered = priority
    .filter((key) => key in params)
    .map((key) => `${key}=${params[key]}`);

  if (ordered.length) {
    return ordered.join(", ");
  }

  return entries
    .slice(0, 4)
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
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
  const contributions = getShapContributions(prediction);
  const visibleContributions = getShapContributionSlice(contributions, currentShapMode);

  if (!prediction || !visibleContributions.length) {
    featureContributions.className = "contribution-box empty-state";
    featureContributions.textContent = currentShapMode === "bottom"
      ? "No five SHAP features pushing away were found for this row."
      : "No SHAP explanation was returned for this row.";
    return;
  }

  const companyLabel = prediction.company_name ? `${prediction.company_name}` : "This company";
  const formattedContributions = visibleContributions.map((item) => ({
    ...item,
    shapValue: Number(item.shap_value) || 0,
    magnitude: Math.abs(Number(item.shap_value) || 0)
  }));
  const maxMagnitude = Math.max(...formattedContributions.map((item) => item.magnitude), 0.000001);
  const items = formattedContributions
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
  const modeLabel = currentShapMode === "bottom" ? "Bottom 5" : "Top 5";
  const summaryCopy = currentShapMode === "bottom"
    ? "The list below shows the five strongest negative features behind that result, ordered from the most negative effect downward."
    : "The list below shows the strongest features behind that result, ordered from biggest effect to smaller effect.";
  featureContributions.innerHTML = `
    <div class="shap-summary">
      <p class="shap-summary-title"><strong>${escapeHtml(companyLabel)}</strong> was predicted as <strong>${escapeHtml(prediction.predicted_rating_group)}</strong>.</p>
      <p class="shap-summary-copy">${escapeHtml(modeLabel)} view. ${escapeHtml(summaryCopy)}</p>
    </div>
    <ul class="shap-list">${items}</ul>
  `;
}

function renderPredictionRows(rows) {
  const body = predictionTable.querySelector("tbody");
  currentPredictions = rows;
  populateShapCompanyOptions(rows);
  currentShapMode = "top";
  updateShapTabState();

  if (!rows || !rows.length) {
    body.innerHTML = '<tr><td colspan="6" class="empty-state">Run an analysis to see prediction rows.</td></tr>';
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
          <td>${escapeHtml(formatClassDistribution(row.class_probabilities))}</td>
          <td>${escapeHtml(row.top_contributions_text || "Unavailable")}</td>
        </tr>`
    )
    .join("");

  const tableRows = body.querySelectorAll("tr[data-row-index]");
  tableRows.forEach((rowElement) => {
    const index = Number(rowElement.dataset.rowIndex);
    const activate = () => {
      setSelectedPredictionRow(index, true);
    };
    rowElement.addEventListener("click", activate);
    rowElement.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });

  currentShapPredictionIndex = 0;
  setSelectedPredictionRow(0, true);
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
  try {
    const payload = collectManualPayload();
    payload.model = manualModelSelect.value;
    setLoadingState(true, submitButton);

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
  metricsNote.textContent = payload.metrics_note || "Model metrics shown here are based on Dataset A's test set, not the uploaded file.";
  renderConfusionMatrix(payload.confusion_matrix, payload.class_labels || []);
  classificationReport.textContent = payload.classification_report_text || "Unavailable";
  renderOverallRiskSummary(payload.predictions || []);
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
      top_contributions: payload.top_feature_contributions,
      shap_contributions: payload.shap_contributions || payload.top_feature_contributions
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
overallRiskSearch.addEventListener("input", () => {
  currentOverallRiskQuery = overallRiskSearch.value || "";
  renderOverallRiskSummaryTable();
});
if (shapCompanySearch) {
  shapCompanySearch.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      showShapForSelectedCompany();
    }
  });
}
if (shapCompanyApply) {
  shapCompanyApply.addEventListener("click", showShapForSelectedCompany);
}
if (shapTopTab) {
  shapTopTab.addEventListener("click", () => setShapMode("top"));
}
if (shapBottomTab) {
  shapBottomTab.addEventListener("click", () => setShapMode("bottom"));
}
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
