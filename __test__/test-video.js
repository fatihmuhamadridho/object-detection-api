const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const ffmpeg = require('fluent-ffmpeg');

async function detectObjectsInVideo(videoPath) {
  // Load the COCO-SSD model
  const model = await cocoSsd.load();

  // Set up the video processing pipeline
  const framesPath = 'frames';
  if (!fs.existsSync(framesPath)) {
    fs.mkdirSync(framesPath);
  }
  const command = ffmpeg(videoPath)
    .output(`${framesPath}/frame-%d.jpg`)
    .on('error', (err) => {
      console.log(`An error occurred while processing the video: ${err.message}`);
    });

  // Process each frame of the video
  command.on('end', async () => {
    const frames = fs.readdirSync(framesPath);
    for (let i = 0; i < frames.length; i++) {
      // Load the frame as a Tensor
      const frameBuffer = fs.readFileSync(`${framesPath}/${frames[i]}`);
      const frameTensor = tf.node.decodeImage(frameBuffer);

      // Run object detection on the frame
      const predictions = await model.detect(frameTensor);

      // Print the predictions to the console
      console.log(`Frame ${i + 1}:`);
      console.log(predictions);
    }
  });

  command.run();
}

detectObjectsInVideo('./public/video.mov');
