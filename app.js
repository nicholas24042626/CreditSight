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
const PREDICT_SCRIPT = path.join(ROOT_DIR, "python", "predict.py");
const MODEL_KEYS = ["decision_tree", "random_forest", "logistic_regression", "xgboost"];
const MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024;

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

const upload = multer({
  storage,
  limits: {
    fileSize: MAX_UPLOAD_SIZE_BYTES
  },
  fileFilter: (_req, file, callback) => {
    const allowedExtensions = [".csv", ".xlsx", ".xls"];
    const extension = path.extname(file.originalname || "").toLowerCase();
    const normalizedName = String(file.originalname || "").toLowerCase();
    const allowedByName = allowedExtensions.some((allowed) => normalizedName.endsWith(allowed));
    const allowedByMime = [
      "text/csv",
      "application/csv",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "text/plain"
    ].includes(file.mimetype);

    if (!allowedByName && !allowedByMime) {
      return callback(new multer.MulterError("LIMIT_UNEXPECTED_FILE", "dataset"));
    }

    if (!allowedExtensions.includes(extension)) {
      return callback(new Error("Unsupported file type. Please upload a CSV or Excel file."));
    }

    return callback(null, true);
  }
});

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

function removeUploadedFile(filePath) {
  if (!filePath) {
    return;
  }

  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (_err) {
    // Upload cleanup should not block the user response.
  }
}

function mapUploadError(error) {
  if (!error) {
    return "Upload failed.";
  }

  if (error instanceof multer.MulterError) {
    if (error.code === "LIMIT_FILE_SIZE") {
      return "File too large. Please upload a file smaller than 10 MB.";
    }
    if (error.code === "LIMIT_UNEXPECTED_FILE") {
      return "Unsupported file type. Please upload a CSV or Excel file.";
    }
    return error.message;
  }

  return error.message || "Upload failed.";
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, name: "CreditSight" });
});

app.post("/api/analyze", (req, res) => {
  // This route receives the uploaded file and sends it to Python.
  upload.single("dataset")(req, res, (uploadError) => {
    if (uploadError) {
      removeUploadedFile(req.file && req.file.path);
      return res.status(400).json({ error: mapUploadError(uploadError) });
    }

    const uploadedPath = req.file && req.file.path;

    try {
      if (!req.file) {
        return res.status(400).json({ error: "Please upload a dataset file before running the analysis." });
      }

      const modelName = String(req.body.model || "").trim();
      if (!MODEL_KEYS.includes(modelName)) {
        return res.status(400).json({
          error: "Invalid model name. Please choose decision_tree, random_forest, logistic_regression, or xgboost."
        });
      }

      const missingModels = MODEL_KEYS.filter((modelKey) => !fs.existsSync(path.join(MODELS_DIR, `${modelKey}.joblib`)));
      if (missingModels.length > 0) {
        return res.status(400).json({
          error: "Trained model files are missing.",
          details: "Run `python train_models.py --data \"path\\to\\set A corporate_rating.csv\"` first to create the saved models."
        });
      }

      const result = runPredictionScript(modelName, uploadedPath);

      if (!result.ok) {
        return res.status(400).json({
          error: result.error,
          details: result.details || null
        });
      }

      const response = result.data;
      if (response.output_csv) {
        response.output_csv_url = `/outputs/${path.basename(response.output_csv)}`;
      }

      return res.json(response);
    } finally {
      removeUploadedFile(uploadedPath);
    }
  });
});

app.use((error, _req, res, next) => {
  if (error instanceof multer.MulterError) {
    return res.status(400).json({ error: mapUploadError(error) });
  }

  if (error) {
    return res.status(400).json({ error: error.message || "Upload failed." });
  }

  return next();
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
