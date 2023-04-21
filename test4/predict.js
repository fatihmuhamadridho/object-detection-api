const tf = require('@tensorflow/tfjs-node');

async function main() {
  const model = await tf.loadLayersModel('file://' + __dirname + '/models/model.json');
  const prediction = model.predict(tf.randomNormal([3, 784]));
  prediction.print();
}

main();
