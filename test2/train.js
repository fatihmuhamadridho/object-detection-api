const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');

async function trainModel() {
  console.log('Loading COCO dataset...');
  const trainData = require('./apple/train/_annotations.coco.json');
  const testData = require('./apple/test/_annotations.coco.json');
  console.log('COCO dataset loaded successfully!');

  console.log('Training model...');
  const model = await cocoSsd.load();
  console.log('Model loaded successfully!');

  const epochs = 10;
  for (let i = 0; i < epochs; i++) {
    console.log(`Epoch ${i + 1}/${epochs}`);
    const history = await model.fit(trainData, testData, { epochs: 1 });
    console.log(`Loss: ${history.history.loss[0]}`);
  }

  console.log('Model trained successfully!');
  return model;
}

async function main() {
  const model = await trainModel();
  // Load image
  const image = await tf.node.decodeImage(
    tf.node.readFileSync(
      './test2/apple/test/purepng-com-green-applesappleapplesfruitsweetgreen-apple-17015271856472meqk_png.rf.7e497773da579b287ac9ffdbfb3ecdfc.jpg'
    )
  );

  // Make prediction
  const predictions = await model.detect(image);

  // Print predictions
  console.log(predictions);
  // Gunakan model untuk deteksi buah apel
}

main();
