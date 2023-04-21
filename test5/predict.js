const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function main() {
  const model = await tf.loadLayersModel('file://' + __dirname + '/models/model.json');
  const data = fs.readFileSync('./adutta_swan.jpg');
  const tensor = tf.node.decodeImage(data);
  const resizedTensor = tf.image.resizeBilinear(tensor, [600, 400]);
  const prediction = model.predict(resizedTensor.expandDims());
  prediction.print();
}

main();
