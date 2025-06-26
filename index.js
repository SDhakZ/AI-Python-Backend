const express = require("express");
const cors = require("cors");
const predictRoutes = require("./routes/predictRoutes");
const trainRoutes = require("./routes/trainRoutes");

const app = express();
const PORT = 5000;

app.use(
  cors({
    origin: "http://localhost:5173",
    credentials: true, // optional, only needed if using cookies/auth headers
  })
);
app.use(express.json());

app.use("/api/predict", predictRoutes);
app.use("/api/train", trainRoutes);

app.listen(PORT, () => {
  console.log(`✅ API running at http://localhost:${PORT}/api/bmi`);
});
