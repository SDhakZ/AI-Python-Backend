const bmiService = require("../services/bmiService");
const bmiMLPService = require("../services/bmiMLPServices");

exports.calculateBMI = async (req, res) => {
  const { height, weight, gender, mode } = req.body;

  if (!height || !weight) {
    return res.status(400).json({ error: "Height and weight are required" });
  }
  if (mode == "mlp") {
    try {
      if (!gender) {
        return res
          .status(400)
          .json({ error: "Gender is required for MLP mode" });
      }
      const bmi = await bmiMLPService.getBMIMLPFromPython(
        height,
        weight,
        gender
      );
      const category = getBMICategory(bmi);

      res.json({
        bmi: parseFloat(bmi.toFixed(2)),
        category,
      });
    } catch (err) {
      console.error("BMI Calculation Error:", err);
      res.status(500).json({ error: "Failed to calculate BMI" });
    }
    return;
  } else if (mode == "linear") {
    try {
      const bmi = await bmiService.getBMIFromPython(height, weight);

      const category = getBMICategory(bmi);

      res.json({
        bmi: parseFloat(bmi.toFixed(2)),
        category,
      });
    } catch (err) {
      console.error("BMI Calculation Error:", err);
      res.status(500).json({ error: "Failed to calculate BMI" });
    }
    return;
  } else {
    return res.status(400).json({ error: "Invalid mode" });
  }
};
// Helper function to classify BMI
function getBMICategory(bmi) {
  if (bmi < 18.5) return "Underweight";
  if (bmi < 25) return "Normal";
  if (bmi < 30) return "Overweight";
  return "Obese";
}
