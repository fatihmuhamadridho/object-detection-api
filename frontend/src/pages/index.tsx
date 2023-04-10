import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

tf.setBackend('webgl')

export default function ObjectDetection() {
  const canvasRef: any = useRef(null);

  useEffect(() => {
    async function runObjectDetection() {
      // Load the COCO-SSD model
      const model = await cocoSsd.load();

      // Get the image URL
      const imageUrl = './images/car.jpg';

      // Load the image data
      const imageElement = document.createElement('img');
      imageElement.src = imageUrl;
      await new Promise(resolve => {
        imageElement.onload = (value) => {
          resolve(value);
        };
      });

      // Get the image dimensions
      const width = imageElement.width;
      const height = imageElement.height;

      // Draw the image on the canvas
      const canvas = canvasRef.current;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imageElement, 0, 0);

      // Perform object detection
      const predictions = await model.detect(canvas);

      // Draw the bounding boxes on the canvas
      predictions.forEach(prediction => {
        ctx.beginPath();
        ctx.lineWidth = '2';
        ctx.strokeStyle = 'red';
        ctx.rect(...prediction.bbox);
        ctx.stroke();
      });
    }

    runObjectDetection();
  }, []);

  return (
    <div>
      <canvas ref={canvasRef}></canvas>
    </div>
  );
}
