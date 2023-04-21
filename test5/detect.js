const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

async function predictImage() {
  // Load the model and metadata
  const modelPath = path.resolve('./test5/models/model.json');
  const model = await tf.loadLayersModel('file://' + modelPath);
  const metadataPath = path.resolve('./test5/models/metadata.json');
  const metadata = JSON.parse(fs.readFileSync(metadataPath));

  // Load an image
  const imagePath = './adutta_swan.jpg';
  const img = fs.readFileSync(imagePath);
  const decodedImg = tf.node.decodeImage(img);

  // Preprocess the image
  const resizedImg = tf.image.resizeBilinear(decodedImg, [metadata.imageSize, metadata.imageSize]);
  const batchedImg = resizedImg.expandDims(0).toFloat().div(127).sub(1);

  // Make predictions
  const predictions = model.predict(batchedImg);
  const topPredictions = Array.from(predictions.dataSync())
    .map((p, i) => ({
      className: metadata.labels[i],
      probability: p
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, metadata.maxPredictions);

  console.log(topPredictions);
}

predictImage();
