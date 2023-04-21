const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// Load dataset
const dataset = require('./coco.json');

// Create arrays to hold images and annotations
const images = [];
const annotations = [];

// Populate images and annotations arrays
dataset.images.forEach((image) => {
  images.push({
    id: image.id,
    width: image.width,
    height: image.height,
    fileName: image.file_name
  });
});

dataset.annotations.forEach((annotation) => {
  annotations.push({
    id: annotation.id,
    imageId: annotation.image_id,
    bbox: annotation.bbox,
    category: annotation.category_id
  });
});

// Create model architecture
const model = tf.sequential({
  layers: [
    tf.layers.flatten({
      inputShape: [600, 400, 3]
    }),
    tf.layers.dense({
      units: 128,
      activation: 'relu'
    }),
    tf.layers.dense({
      units: 4,
      activation: 'sigmoid'
    })
  ]
});

// Compile model
model.compile({
  optimizer: 'adam',
  loss: 'meanSquaredError'
});

// Train model
const xs = [];
const ys = [];

images.forEach((image) => {
  const data = fs.readFileSync(`./images/${image.fileName}`);
  const tensor = tf.node.decodeImage(data);
  const resizedTensor = tf.image.resizeNearestNeighbor(tensor, [600, 400]);
  xs.push(resizedTensor);
  const annotationsForImage = annotations.filter((annotation) => annotation.imageId === image.id);
  annotationsForImage.forEach((annotation) => {
    ys.push(annotation.bbox);
  });
});

console.log(xs, ys);
const xTensor = tf.stack(xs);
const yTensor = tf.stack(ys);

model.fit(xTensor, yTensor, { epochs: 10 }).then(() => {
  // Save model
  model.save('file://' + __dirname + '/models').then(() => {
    console.log('Model saved.');
  });
});
