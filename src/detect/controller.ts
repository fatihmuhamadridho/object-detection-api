const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const ffmpeg = require('fluent-ffmpeg');

class DetectController {
  static async index() {
    return {
      status: true,
      data: null
    };
  }
  static async show() {}
  static create = {
    uploadImage: async (imagePath: any) => {
      // Load the COCO-SSD model
      const model = await cocoSsd.load();

      // Load the image as a Tensor
      const imageBuffer = fs.readFileSync(imagePath);
      const imageTensor = tf.node.decodeImage(imageBuffer);

      // Run object detection on the image
      const predictions = await model.detect(imageTensor);

      // Print the predictions
      // console.log('Predictions:');
      // console.log(predictions);
      return predictions;
    },
    uploadVideo: async (videoPath: any) => {
      const result: any = [];

      // Load the COCO-SSD model
      const model = await cocoSsd.load();

      // Set up the video processing pipeline
      const framesPath = 'frames';
      if (!fs.existsSync(framesPath)) {
        fs.mkdirSync(framesPath);
      }
      const command = ffmpeg(videoPath)
        .output(`${framesPath}/frame-%d.jpg`)
        .on('error', (err: any) => {
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
          result.push(predictions);

          // Print the predictions to the console
          console.log(`Frame ${i + 1}:`);
          console.log(predictions);
        }
      });

      command.run();
      // return result;
    }
  };
  static async update() {}
  static async delete() {}
}

export { DetectController };
