const bmiService = require("../services/bmiService");

exports.calculateBMI = async (req, res) => {
  const { height, weight } = req.body;

  if (!height || !weight) {
    return res.status(400).json({ error: "Height and weight are required" });
  }

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
};

// Helper function to classify BMI
function getBMICategory(bmi) {
  if (bmi < 18.5) return "Underweight";
  if (bmi < 25) return "Normal";
  if (bmi < 30) return "Overweight";
  return "Obese";
}
