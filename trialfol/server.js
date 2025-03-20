// server.js
const express = require("express");
const { google } = require("googleapis");
const cors = require("cors");
const fs = require("fs");
const axios = require("axios");
const path = require("path");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.static("public"));

// Initialize the Google API client
const oauth2Client = new google.auth.OAuth2(
  process.env.CLIENT_ID,
  process.env.CLIENT_SECRET,
  process.env.REDIRECT_URI
);

// Generate the URL for Google authentication
app.get("/auth/google", (req, res) => {
    const scopes = [
        "https://www.googleapis.com/auth/fitness.activity.read", // Step count, activities, workout tracking
        "https://www.googleapis.com/auth/fitness.activity.write", // Write activity data
        "https://www.googleapis.com/auth/fitness.body.read", // Body metrics (weight, height, BMI)
        "https://www.googleapis.com/auth/fitness.body.write", // Write body metrics
        "https://www.googleapis.com/auth/fitness.heart_rate.read", // Heart rate tracking
        "https://www.googleapis.com/auth/fitness.heart_rate.write", // Write heart rate data
        "https://www.googleapis.com/auth/fitness.blood_pressure.read", // Blood pressure data
        "https://www.googleapis.com/auth/fitness.blood_pressure.write", // Write blood pressure data
        "https://www.googleapis.com/auth/fitness.body_temperature.read", // Body temperature data
        "https://www.googleapis.com/auth/fitness.oxygen_saturation.read", // Blood oxygen (SpO2) levels
        "https://www.googleapis.com/auth/fitness.location.read", // GPS/location tracking
        "https://www.googleapis.com/auth/fitness.nutrition.read", // Nutrition tracking
        "https://www.googleapis.com/auth/fitness.nutrition.write", // Write nutrition data
        "https://www.googleapis.com/auth/fitness.sleep.read", // Sleep tracking
        "https://www.googleapis.com/auth/fitness.sleep.write", // Write sleep data
    ];
    
    

  const url = oauth2Client.generateAuthUrl({
    access_type: "offline",
    scope: scopes,
  });
  res.redirect(url);
});

// Callback route
app.get("/auth/google/callback", async (req, res) => {
  const { code } = req.query;
  const { tokens } = await oauth2Client.getToken(code);
  oauth2Client.setCredentials(tokens);
  console.log("/auth/google/callback");
  fs.writeFileSync("token.json", JSON.stringify(tokens), (err) => {
    if (err) {
      console.error("Error saving token:", err);
    } else {
      console.log("Token saved to token.json");
    }
  });
  await fetchGoogleFitData(tokens.access_token);
  res.redirect("/dashboard"); // Redirect to dashboard after successful authentication
});
app.get("/dashboard", (req, res) => {
  res.sendFile(path.join(__dirname, "public/a.html"));
});
// Route to serve the HTML page
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public/index.html"));
});

// Load access token from token.json for fetching data
let accessToken;
try {
  const tokenData = JSON.parse(fs.readFileSync("token.json", "utf8"));
  accessToken = tokenData.access_token;
  console.log(`accessToken read...`);
} catch (err) {
  console.error("Error loading token:", err);
}
// Real-time data connection (SSE)
app.get("/events", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  
  const sendData = () => {
    // Send the latest Google Fit data to the client
    const filePath = path.join(__dirname, "google_fit_data.json");

    if (fs.existsSync(filePath)) {
      try {
        const data = fs.readFileSync(filePath, "utf8");
        res.write(`data: ${data}\n\n`);
      } catch (err) {
        console.error("Error sending data:", err);
      }
    }
  };

  // Send data every 5 seconds (or adjust interval)
  const intervalId = setInterval(sendData, 5000);

  // Clean up when the connection closes
  req.on("close", () => {
    clearInterval(intervalId);
  });
});


const startOfDay = new Date();
startOfDay.setHours(0, 0, 0, 0);
// Route to fetch Google Fit data
async function fetchGoogleFitData(accessToken) {
  try {
    const response = await axios.post(
      "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate",
      {
        aggregateBy: [
          { dataTypeName: "com.google.step_count.delta" },
          { dataTypeName: "com.google.calories.expended" },
          { dataTypeName: "com.google.distance.delta" },
          { dataTypeName: "com.google.activity.segment" },
          { dataTypeName: "com.google.heart_rate.bpm" },
          { dataTypeName: "com.google.body.temperature" },
          { dataTypeName: "com.google.blood_pressure" }, // New feature
          { dataTypeName: "com.google.oxygen_saturation" },
          { dataTypeName: "com.google.nutrition" },
          { dataTypeName: "com.google.sleep.segment" },
        ],
        bucketByTime: {
          durationMillis: 30000, // Daily data
        },
        startTimeMillis: startOfDay.getTime(), // morning
        endTimeMillis: Date.now(),
      },
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      }
    );
    const fetchedData = response.data;
    // Save data to a JSON file
    console.log("Google Fit data fetched successfully001");

    fs.writeFileSync(
      "google_fit_data.json",
      JSON.stringify(fetchedData),
      (err) => {
        if (err) {
          console.error("Error saving Google Fit data:", err);
        } else {
          console.log("Google Fit data saved to google_fit_data.json");
        }
      }
    );

    const historyFilePath = "google_fit_history.json";
    let historyData = [];
    if (fs.existsSync(historyFilePath)) {
      historyData = JSON.parse(fs.readFileSync(historyFilePath, "utf8"));
    }
    historyData.push({ timestamp: new Date(), data: fetchedData });
    fs.writeFileSync(historyFilePath, JSON.stringify(historyData), (err) => {
      if (err) console.error("Error saving history data:", err);
    });
  } catch (error) {
    console.error(
      "Error fetching Google Fit data:",
      error.response ? error.response.data : error.message
    );
  }
}
// Polling function for real-time updates
setInterval(() => {
  if (accessToken) {
    fetchGoogleFitData(accessToken);
  }
}, 30000); // Fetch data every 30 sec

// Route to retrieve saved Google Fit data
app.get("/retrieve-google-fit-data", (req, res) => {
  const filePath = path.join(__dirname, "google_fit_data.json");

  // Check if the file exists before reading
  if (fs.existsSync(filePath)) {
    try {
      const data = fs.readFileSync(filePath, "utf8");
      res.json(JSON.parse(data));
    } catch (err) {
      console.error("Error retrieving Google Fit data:", err);
      res.status(500).json({ error: "Failed to retrieve Google Fit data" });
    }
  } else {
    // If the file doesn't exist, send a message or return empty data
    res.json({ message: "No Google Fit data available yet." });
  }
});
// Route to retrieve Google Fit history data
app.get("/retrieve-google-fit-history", (req, res) => {
  const filePath = path.join(__dirname, "google_fit_history.json");

  // Check if the file exists before reading
  if (fs.existsSync(filePath)) {
    try {
      const data = fs.readFileSync(filePath, "utf8");
      res.json(JSON.parse(data));
    } catch (err) {
      console.error("Error retrieving Google Fit history:", err);
      res.status(500).json({ error: "Failed to retrieve Google Fit history" });
    }
  } else {
    res.json({ message: "No Google Fit history data available yet." });
  }
});
// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
