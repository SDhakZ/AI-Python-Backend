const { spawn } = require("child_process");
const path = require("path");

exports.getXorGateFromPython = (a, b) => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, "../python/predict_xor_gate.py");

    const process = spawn("python", [scriptPath, a, b]);
    let result = "";
    process.stdout.on("data", (data) => {
      result += data.toString();
    });
    process.stderr.on("data", (data) => {
      console.error("Python error:", data.toString());
    });
    process.on("close", (code) => {
      const lines = result.trim().split("\n");
      const lastLine = lines[lines.length - 1];
      const output = parseInt(lastLine);

      if (isNaN(output)) {
        reject("Invalid value from Python script");
      } else {
        resolve(output);
      }
    });
  });
};
