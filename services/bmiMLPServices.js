const { spawn } = require("child_process");
const path = require("path");

exports.getBMIMLPFromPython = (height, weight, gender) => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(
      __dirname,
      "../python/predict/predict_bmi_mlp.py"
    );
    const process = spawn("python3", [scriptPath, height, weight, gender]);
    let result = "";

    process.stdout.on("data", (data) => {
      result += data.toString();
    });

    process.stderr.on("data", (data) => {
      console.error("Python error:", data.toString());
    });

    process.on("close", (code) => {
      const bmi = parseFloat(result);
      if (isNaN(bmi)) {
        reject("Invalid BMI value from Python script");
      } else {
        resolve(bmi);
      }
    });
  });
};
