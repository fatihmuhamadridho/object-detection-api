const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// Load the saved model
const modelPath = './public/helm/helm.json';
const model = await tf.loadGraphModel(`file://${modelPath}`);

// Load the image to predict
const imagePath = 'path/to/image.jpg';
const imageBuffer = fs.readFileSync(imagePath);
const tfImage = tf.node.decodeImage(imageBuffer);

// Resize the image to the expected input shape of the model
const resized = tf.image.resizeBilinear(tfImage, [224, 224]);

// Normalize the pixel values of the image
const offset = tf.scalar(127.5);
const normalized = resized.sub(offset).div(offset);

// Make the prediction using the loaded model
const predictions = await model.executeAsync(normalized.expandDims());

// Get the predicted class and bounding box coordinates
const [boxes, classes, scores] = predictions;
const classIndex = classes.dataSync()[0];
const className = metadata.classes[classIndex];
const [yMin, xMin, yMax, xMax] = boxes.dataSync();
const boundingBox = { yMin, xMin, yMax, xMax };

console.log(`Predicted class: ${className}`);
console.log(`Bounding box: ${JSON.stringify(boundingBox)}`);
