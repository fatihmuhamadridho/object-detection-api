const coco = require('coco-ssd');
const tf = require('@tensorflow/tfjs-node');
const tfconverter = require('@tensorflow/tfjs-converter');
const fs = require('fs');

// Load COCO SSD model
const model = await coco.load();

// Convert the model to TensorFlow.js format
const tfjsModel = await tfconverter.convert(model, { inputSize: 416, outputStride: 32 });

// Save the converted model
fs.writeFileSync('model.json', JSON.stringify(tfjsModel));

// Save the model weights
await tfjsModel.save('file://model');
