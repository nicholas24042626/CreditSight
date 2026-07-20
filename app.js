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
const MAX_PYTHON_BUFFER_BYTES = Number(process.env.PYTHON_MAX_BUFFER_BYTES || 50 * 1024 * 1024);

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

function runPredictionScript(modelName, inputPath, mapping = null) {
  // The Python script does the real machine learning work.
  const resultFileName = `creditsight_${modelName}_${Date.now()}.csv`;
  const outputPath = path.join(OUTPUT_DIR, resultFileName);
  const tempMappingPath = mapping ? path.join(UPLOAD_DIR, `mapping_${Date.now()}_${Math.random().toString(16).slice(2)}.json`) : null;

  if (tempMappingPath) {
    fs.writeFileSync(tempMappingPath, JSON.stringify(mapping, null, 2), "utf-8");
  }

  try {
    const pythonArgs = ["--model", modelName, "--input", inputPath, "--output", outputPath];
    if (tempMappingPath) {
      pythonArgs.push("--mapping-file", tempMappingPath);
    }
    const pythonResult = spawnSync(
      PYTHON_BIN,
      [PREDICT_SCRIPT, ...pythonArgs],
      {
        encoding: "utf-8",
        maxBuffer: MAX_PYTHON_BUFFER_BYTES
      }
    );

    if (pythonResult.error) {
      if (pythonResult.error.code === "ENOBUFS") {
        return {
          ok: false,
          error: "The prediction output exceeded Node's buffer limit.",
          details: `Increase PYTHON_MAX_BUFFER_BYTES or reduce the number of returned rows. Current limit: ${MAX_PYTHON_BUFFER_BYTES} bytes.`
        };
      }

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
  } finally {
    removeUploadedFile(tempMappingPath);
  }
}

function runCompatibilityScript(modelName, inputPath) {
  const pythonResult = spawnSync(
    PYTHON_BIN,
    [PREDICT_SCRIPT, "--model", modelName, "--input", inputPath, "--output", path.join(OUTPUT_DIR, "_compatibility.csv"), "--compatibility"],
    {
      encoding: "utf-8",
      maxBuffer: MAX_PYTHON_BUFFER_BYTES
    }
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
        error: (pythonResult.stderr || pythonResult.stdout || "Compatibility check failed").trim()
      };
    }

    return {
      ok: false,
      error: parsedError.error || "Compatibility check failed",
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

function ensureModelArtifactsExist() {
  const missingModels = MODEL_KEYS.filter((modelKey) => !fs.existsSync(path.join(MODELS_DIR, `${modelKey}.joblib`)));
  if (missingModels.length > 0) {
    return {
      ok: false,
      error: "Trained model files are missing.",
      details: "Run `python train_models.py --data \"path\\to\\set A corporate_rating.csv\"` first to create the saved models."
    };
  }

  return { ok: true };
}

function validateModelName(modelName) {
  const normalized = String(modelName || "").trim();
  if (!MODEL_KEYS.includes(normalized)) {
    return null;
  }
  return normalized;
}

function writeManualInputFile(record) {
  const fileName = `manual_${Date.now()}_${Math.random().toString(16).slice(2)}.json`;
  const manualPath = path.join(UPLOAD_DIR, fileName);
  fs.writeFileSync(manualPath, JSON.stringify([record], null, 2), "utf-8");
  return manualPath;
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

      const modelName = validateModelName(req.body.model);
      if (!modelName) {
        return res.status(400).json({
          error: "Invalid model name. Please choose decision_tree, random_forest, logistic_regression, or xgboost."
        });
      }

      const modelCheck = ensureModelArtifactsExist();
      if (!modelCheck.ok) {
        return res.status(400).json({
          error: modelCheck.error,
          details: modelCheck.details
        });
      }

      let mapping = null;
      if (req.body.mapping) {
        try {
          mapping = JSON.parse(req.body.mapping);
        } catch (_err) {
          return res.status(400).json({
            error: "The submitted mapping payload is invalid JSON."
          });
        }
      }

      const result = runPredictionScript(modelName, uploadedPath, mapping);

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

app.post("/api/compatibility", (req, res) => {
  upload.single("dataset")(req, res, (uploadError) => {
    if (uploadError) {
      removeUploadedFile(req.file && req.file.path);
      return res.status(400).json({ error: mapUploadError(uploadError) });
    }

    const uploadedPath = req.file && req.file.path;

    try {
      if (!req.file) {
        return res.status(400).json({ error: "Please upload a dataset file before checking compatibility." });
      }

      const modelName = validateModelName(req.body.model);
      if (!modelName) {
        return res.status(400).json({
          error: "Invalid model name. Please choose decision_tree, random_forest, logistic_regression, or xgboost."
        });
      }

      const modelCheck = ensureModelArtifactsExist();
      if (!modelCheck.ok) {
        return res.status(400).json({
          error: modelCheck.error,
          details: modelCheck.details
        });
      }

      const result = runCompatibilityScript(modelName, uploadedPath);
      if (!result.ok) {
        return res.status(400).json({
          error: result.error,
          details: result.details || null
        });
      }

      return res.json(result.data);
    } finally {
      removeUploadedFile(uploadedPath);
    }
  });
});

app.post("/api/analyze-manual", (req, res) => {
  try {
    const modelName = validateModelName(req.body && req.body.model);
    if (!modelName) {
      return res.status(400).json({
        error: "Invalid model name. Please choose decision_tree, random_forest, logistic_regression, or xgboost."
      });
    }

    const modelCheck = ensureModelArtifactsExist();
    if (!modelCheck.ok) {
      return res.status(400).json({
        error: modelCheck.error,
        details: modelCheck.details
      });
    }

    if (!req.body || typeof req.body !== "object") {
      return res.status(400).json({ error: "Manual company assessment data is missing." });
    }

    const tempInputPath = writeManualInputFile(req.body);
    try {
      const result = runPredictionScript(modelName, tempInputPath);

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
      removeUploadedFile(tempInputPath);
    }
  } catch (error) {
    return res.status(400).json({ error: error.message || "Manual analysis failed." });
  }
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
