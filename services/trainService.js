const { spawn } = require("child_process");
const path = require("path");

exports.runTraining = () => {
  return new Promise((resolve, reject) => {
    const trainScript = spawn("python", [
      path.join(__dirname, "../python/train_model.py"),
    ]);

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
