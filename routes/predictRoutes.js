const express = require("express");
const router = express.Router();
const bmiController = require("../controllers/bmiController");
const gateController = require("../controllers/gateController");

router.post("/bmi", bmiController.calculateBMI);
router.post("/perceptronGate", gateController.calculateGate);

module.exports = router;
