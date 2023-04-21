const tf = require('@tensorflow/tfjs-node');

const model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
    tf.layers.dense({ units: 10, activation: 'softmax' })
  ]
});

model.weights.forEach((w) => {
  console.log(w.name, w.shape);
});

model.weights.forEach((w) => {
  const newVals = tf.randomNormal(w.shape);
  // w.val is an instance of tf.Variable
  w.val.assign(newVals);
});

model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// // Generate dummy data.
// const data = tf.randomNormal([100, 784]);
// const labels = tf.randomUniform([100, 10]);

// function onBatchEnd(batch, logs) {
//   console.log('Accuracy', logs.acc);
// }

// // Train for 5 epochs with batch size of 32.
// model
//   .fit(data, labels, {
//     epochs: 5,
//     batchSize: 32,
//     callbacks: { onBatchEnd }
//   })
//   .then((info) => {
//     console.log('Final accuracy', info.history.acc);
//   });

function* data() {
  for (let i = 0; i < 100; i++) {
    // Generate one sample at a time.
    yield tf.randomNormal([784]);
  }
}

function* labels() {
  for (let i = 0; i < 100; i++) {
    // Generate one sample at a time.
    yield tf.randomUniform([10]);
  }
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({ xs, ys }).shuffle(100 /* bufferSize */).batch(32);

// Train the model for 5 epochs.
model.fitDataset(ds, { epochs: 5 }).then((info) => {
  console.log('Accuracy', info.history.acc);
});

// Predict 3 random samples.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();

const saveResult = async () => {
  await model.save('file://' + __dirname + '/models');
};

saveResult();
