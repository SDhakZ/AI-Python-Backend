const trainService = require("../services/trainService");

exports.trainModel = async (req, res) => {
  try {
    await trainService.runTraining();
    res.status(200).json({ message: "Model trained and saved successfully." });
  } catch (err) {
    console.error("Training Error:", err);
    res.status(500).json({ error: "Failed to train the model." });
  }
};
