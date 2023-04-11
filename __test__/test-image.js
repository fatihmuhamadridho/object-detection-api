const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');

async function detectObjectsInImage(imagePath) {
  // Load the COCO-SSD model
  const model = await cocoSsd.load();

  // Load the image as a Tensor
  const imageBuffer = fs.readFileSync(imagePath);
  const imageTensor = tf.node.decodeImage(imageBuffer);

  // Run object detection on the image
  const predictions = await model.detect(imageTensor);

  // Print the predictions
  console.log('Predictions:');
  console.log(predictions);
}

detectObjectsInImage('./public/helm/helm.jpg');
