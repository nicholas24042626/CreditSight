const express = require("express");
const fs = require("fs");
const path = require("path");
const multer = require("multer");
const { spawnSync } = require("child_process");

const app = express();
const PORT = process.env.PORT || 3000;
const ROOT_DIR = __dirname;
const PUBLIC_DIR = path.join(ROOT_DIR, "public");
const UPLOAD_DIR = path.join(ROOT_DIR, "uploads");
const OUTPUT_DIR = path.join(ROOT_DIR, "outputs");
const MODELS_DIR = path.join(ROOT_DIR, "models");
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const TRAIN_SCRIPT = path.join(ROOT_DIR, "python", "train_models.py");
const PREDICT_SCRIPT = path.join(ROOT_DIR, "python", "predict.py");
const MODEL_KEYS = ["decision_tree", "random_forest", "logistic_regression", "xgboost"];

// Make sure the folders we write to always exist.
for (const dir of [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (_req, _file, callback) => callback(null, UPLOAD_DIR),
  filename: (_req, file, callback) => {
    const safeName = `${Date.now()}_${path.basename(file.originalname)}`;
    callback(null, safeName);
  }
});

const upload = multer({ storage });

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use("/outputs", express.static(OUTPUT_DIR));
app.use(express.static(PUBLIC_DIR));

function runPredictionScript(modelName, inputPath) {
  // The Python script does the real machine learning work.
  const resultFileName = `creditsight_${modelName}_${Date.now()}.csv`;
  const outputPath = path.join(OUTPUT_DIR, resultFileName);

  const pythonResult = spawnSync(
    PYTHON_BIN,
    [PREDICT_SCRIPT, "--model", modelName, "--input", inputPath, "--output", outputPath],
    { encoding: "utf-8" }
  );

  if (pythonResult.error) {
    return {
      ok: false,
      error: pythonResult.error.message
    };
  }

  if (pythonResult.status !== 0) {
    let parsedError = null;
    try {
      parsedError = JSON.parse((pythonResult.stderr || pythonResult.stdout || "").trim());
    } catch (_err) {
      parsedError = {
        error: (pythonResult.stderr || pythonResult.stdout || "Python script failed").trim()
      };
    }

    return {
      ok: false,
      error: parsedError.error || "Prediction failed",
      details: parsedError.details || null
    };
  }

  try {
    return {
      ok: true,
      data: JSON.parse((pythonResult.stdout || "").trim())
    };
  } catch (_err) {
    return {
      ok: false,
      error: "Python script returned invalid JSON."
    };
  }
}

function modelArtifactsExist() {
  return MODEL_KEYS.every((modelKey) => fs.existsSync(path.join(MODELS_DIR, `${modelKey}.joblib`)));
}

function runTrainingScript(inputPath) {
  const pythonResult = spawnSync(
    PYTHON_BIN,
    [TRAIN_SCRIPT, "--data", inputPath],
    { encoding: "utf-8" }
  );

  if (pythonResult.error) {
    return {
      ok: false,
      error: pythonResult.error.message
    };
  }

  if (pythonResult.status !== 0) {
    let parsedError = null;
    try {
      parsedError = JSON.parse((pythonResult.stderr || pythonResult.stdout || "").trim());
    } catch (_err) {
      parsedError = {
        error: (pythonResult.stderr || pythonResult.stdout || "Training script failed").trim()
      };
    }

    return {
      ok: false,
      error: parsedError.error || "Training failed",
      details: parsedError.details || null
    };
  }

  try {
    return {
      ok: true,
      data: JSON.parse((pythonResult.stdout || "").trim())
    };
  } catch (_err) {
    return {
      ok: false,
      error: "Training script returned invalid JSON."
    };
  }
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, name: "CreditSight" });
});

app.post("/api/analyze", upload.single("dataset"), (req, res) => {
  // This route receives the uploaded file and sends it to Python.
  if (!req.file) {
    return res.status(400).json({ error: "Please upload a dataset file before running the analysis." });
  }

  const modelName = String(req.body.model || "").trim();
  if (!modelName) {
    return res.status(400).json({ error: "Please choose one model from the dropdown menu." });
  }

  let trainingSummary = null;
  if (!modelArtifactsExist()) {
    const trainingResult = runTrainingScript(req.file.path);
    if (!trainingResult.ok) {
      return res.status(400).json({
        error: trainingResult.error,
        details: trainingResult.details || null
      });
    }
    trainingSummary = trainingResult.data;
  }

  const result = runPredictionScript(modelName, req.file.path);

  if (!result.ok) {
    return res.status(400).json({
      error: result.error,
      details: result.details || null
    });
  }

  const response = result.data;
  if (trainingSummary) {
    response.training_summary = trainingSummary;
  }
  if (response.output_csv) {
    response.output_csv_url = `/outputs/${path.basename(response.output_csv)}`;
  }

  return res.json(response);
});

app.use((req, res) => {
  if (req.accepts("html")) {
    return res.sendFile(path.join(PUBLIC_DIR, "index.html"));
  }
  return res.status(404).json({ error: "Not found" });
});

app.listen(PORT, () => {
  console.log(`CreditSight is running at http://localhost:${PORT}`);
});
