const gateService = require("../services/gateServices");
const xorGateServices = require("../services/xorGateServices");
exports.calculateGate = async (req, res) => {
  const { a, b } = req.body;
  const { gate } = req.query;
  console.log("this is gate", String(gate));
  if (a === undefined || b === undefined) {
    return res.status(400).json({ error: "Both a and b are required" });
  }

  try {
    gateResult = null;
    if (gate == "and" || gate == "or") {
      gateResult = await gateService.getGateFromPython(a, b, gate);
    } else if (gate == "xor") {
      gateResult = await xorGateServices.getXorGateFromPython(a, b);
    } else {
      return res.status(400).json({ error: "Invalid gate type" });
    }

    res.json({ output: Number(gateResult) });
  } catch (err) {
    console.error("Gate Calculation Error:", err);
    res.status(500).json({ error: "Failed to calculate Gate" });
  }
};
