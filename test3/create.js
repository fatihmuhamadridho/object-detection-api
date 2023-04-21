const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function train() {
  const csvFilePath = 'vgg.csv';

  // Load annotations from CSV file
  const csvData = fs.readFileSync(csvFilePath, 'utf-8');
  const lines = csvData.trim().split('\n');
  const headers = lines[0].split(',');
  const annotations = [];

  for (let i = 1; i < lines.length; i++) {
    const data = lines[i].split(',');
    const annotation = {};
    for (let j = 0; j < headers.length; j++) {
      if (headers[j] === 'region_shape_attributes') {
        annotation[headers[j]] = JSON.parse(data[j]);
      } else if (headers[j] === 'region_attributes') {
        annotation[headers[j]] = JSON.parse(data[j]);
      } else {
        annotation[headers[j]] = data[j];
      }
    }
    annotations.push(annotation);
  }

  console.log('testtesttesttesttesttesttesttesttesttesttesttest');
  // Prepare data for training
  const xs = [];
  const ys = [];

  for (const annotation of annotations) {
    const imgPath = annotation.filename;
    const imgData = fs.readFileSync(imgPath);
    const imgTensor = tf.node.decodeImage(imgData);
    xs.push(imgTensor);

    const regions = annotation.region_shape_attributes;
    const labels = annotation.region_attributes;
    const boxes = [];
    const classes = [];

    for (let i = 0; i < regions.all_points_x?.length; i++) {
      const x = regions.all_points_x[i];
      const y = regions.all_points_y[i];
      boxes.push([x, y, x, y]);
      classes.push(labels.name);
    }

    const labelTensor = tf.tidy(() => {
      const boxesTensor = tf.tensor2d(boxes);
      const classesTensor = tf.tensor1d(classes, 'string');
      return { boxes: boxesTensor, classes: classesTensor };
    });

    ys.push(labelTensor);
  }

  const xsTensor = tf.stack(xs);
  const ysTensor = tf.data.array(ys);

  const dataset = tf.data.zip({ xs: xsTensor, ys: ysTensor }).shuffle(annotations.length).batch(32);

  // Define and compile model
  const baseModel = await tf.loadGraphModel('model.json');
  const outputLayer = baseModel.layers[1].output;
  const model = tf.model({ inputs: baseModel.inputs, outputs: outputLayer });

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  // Train model
  await model.fitDataset(dataset, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
      }
    }
  });

  // Save model
  await model.save('trained_model');
}

train();
