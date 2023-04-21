const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const jpeg = require('jpeg-js');

const MODEL_PATH =
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
const IMAGE_PATH = './test2/apple.jpg';
const CLASS_NAMES = ['apple', 'banana', 'orange', 'pear', 'kiwi'];

async function run() {
  const model = await tf.loadLayersModel(MODEL_PATH);

  const imageBuffer = fs.readFileSync(IMAGE_PATH);
  const imageData = jpeg.decode(imageBuffer, true);
  const input = tf.browser.fromPixels(imageData).resizeNearestNeighbor([224, 224]).toFloat();
  const meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);
  const processedInput = input.sub(meanImageNetRGB).reverse(2);

  const preds = model.predict(processedInput.reshape([-1, 224, 224, 3]));
  const topPreds = Array.from(preds.dataSync())
    .map((p, i) => ({ probability: p, className: CLASS_NAMES[i] }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);

  console.log(topPreds);
}

run();
