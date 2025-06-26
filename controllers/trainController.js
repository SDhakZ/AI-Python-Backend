// controllers/trainController.js
const trainService = require("../services/trainService");

exports.trainModel = async (req, res) => {
  const { modelType } = req.body;

  try {
    await trainService.runTraining(modelType);
    res
      .status(200)
      .json({ message: `Model '${modelType}' trained successfully.` });
  } catch (err) {
    console.error("Training Error:", err);
    res.status(500).json({ error: "Failed to train the model." });
  }
};
