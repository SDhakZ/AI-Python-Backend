const express = require("express");
const cors = require("cors");
const bmiRoutes = require("./routes/bmiRoutes");
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

app.use("/api", bmiRoutes);
app.use("/api", trainRoutes);

app.listen(PORT, () => {
  console.log(`✅ API running at http://localhost:${PORT}/api/bmi`);
});
