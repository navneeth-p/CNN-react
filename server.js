// server.js (Express backend)
process.env['TF_CPP_MIN_LOG_LEVEL'] = '2';
const fs = require('fs');
const express = require('express');
const multer = require('multer');
const { createCanvas, Image } = require('canvas');
const { decode } = require('base64-arraybuffer');
const tf = require('@tensorflow/tfjs');
const { loadGraphModel } = require('@tensorflow/tfjs-converter');
const tfnode = require('@tensorflow/tfjs-node');
const cors = require("cors");

const app = express();
app.use(cors());

let model;
// const modelPath = './static/model-pickle.pkl';



async function loadModel() {
  // const modelData = fs.readFileSync(modelPath);
  // const  model = await tf.node.loadGraphModel(modelData, ['serve'], 'tensorflow');
  const model = await loadGraphModel('file://static/model.model');
  }
  
  loadModel().catch((err) => {
    console.error('Error loading the model:', err);
  });

function preprocess(image) {
  const canvas = createCanvas(100, 100);
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.src = image;
  ctx.drawImage(img, 0, 0, 100, 100);
  const preprocessed = tf.browser.fromPixels(canvas).reshape([1, 100, 100, 1]).toFloat().div(255.0);
  return preprocessed;
}

app.use(express.json());

// Configure multer for file upload
const storage = multer.memoryStorage();
const upload = multer({ storage });

app.get('/', (req, res) => {
  res.send('Hello, World!22');
});

// app.post('/api/predict', (req, res) => {
//     res.send('/predict Works');
//   });

app.post('/api/predict', upload.single('image'), async (req, res) => {
  const encoded = req.file.buffer.toString('base64');
  const decoded = decode(encoded);
  const preprocessedImage = preprocess(decoded);

  const prediction = await model.predict(preprocessedImage);
  const result = prediction.argMax(1).dataSync()[0];
  const accuracy = prediction.max().dataSync()[0];
  const label = result === 0 ? 'Covid19 Negative' : 'Covid19 Positive';

  const response = { prediction: { result: label, accuracy: accuracy } };

  res.json(response);
});

const port = 3001; // Replace with your desired port number
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
