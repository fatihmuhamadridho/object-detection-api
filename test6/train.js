const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

async function main() {
  // Create MobileNet-v2 model for transfer learning
  const NUM_CLASSES = 2;
  const baseModel = await tf.loadLayersModel(
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/1/default/1',
    { fromTFHub: true }
  );
  const layer = baseModel.layers[0];
  const newOutputLayer = tf.layers.dense({
    units: NUM_CLASSES,
    activation: 'softmax',
    name: 'Output'
  });
  const model = tf.model({
    inputs: layer.inboundNodes[0].input,
    outputs: newOutputLayer.apply(layer.output)
  });

  // Load COCO-annotator dataset
  const dataset = JSON.parse(fs.readFileSync('coco.json', 'utf8'));

  // Prepare dataset
  const images = dataset.images;
  const annotations = dataset.annotations;
  const labels = annotations.map((annotation) => (annotation.category_id === 1 ? 1 : 0));

  // Load and preprocess images for model input
  const imageTensors = await Promise.all(
    images.map(async (image) => {
      const img = await loadImage(image.file_name);
      const canvas = createCanvas(img.width, img.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      const tensor = tf.browser
        .fromPixels(imageData)
        .resizeBilinear([224, 224])
        .toFloat()
        .div(tf.scalar(255));
      const expanded = tensor.expandDims(0);
      return expanded;
    })
  );

  // Train model
  const batchSize = 8;
  const epochs = 10;
  const optimizer = tf.train.adam(0.0001);
  model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  const history = await model.fit(tf.data.zip({ xs: imageTensors }), tf.tensor(labels), {
    batchSize: batchSize,
    epochs: epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1} loss: ${logs.loss.toFixed(4)}, accuracy: ${logs.acc.toFixed(4)}`
        );
      }
    }
  });

  // Save trained model
  await model.save('apple_detection_model');
  console.log('Model saved');
}

main();
