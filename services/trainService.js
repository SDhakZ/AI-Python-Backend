// services/trainService.js
const { spawn } = require("child_process");
const path = require("path");

exports.runTraining = (modelType) => {
  return new Promise((resolve, reject) => {
    let scriptPath;

    switch (modelType) {
      case "and":
        scriptPath = "../python/train_and.py";
        break;
      case "or":
        scriptPath = "../python/train_or.py";
        break;
      case "bmi":
        scriptPath = "../python/train_bmi_model.py";
        break;
      default:
        return reject(new Error("Unknown model type"));
    }

    const trainScript = spawn("python", [path.join(__dirname, scriptPath)]);

    trainScript.stdout.on("data", (data) => {
      console.log(`Training output: ${data}`);
    });

    trainScript.stderr.on("data", (data) => {
      console.error(`Training error: ${data}`);
    });

    trainScript.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Training process exited with code ${code}`));
      }
    });
  });
};
