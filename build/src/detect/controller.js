"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var _a;
Object.defineProperty(exports, "__esModule", { value: true });
exports.DetectController = void 0;
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const ffmpeg = require('fluent-ffmpeg');
class DetectController {
    static index() {
        return __awaiter(this, void 0, void 0, function* () {
            return {
                status: true,
                data: null
            };
        });
    }
    static show() {
        return __awaiter(this, void 0, void 0, function* () { });
    }
    static update() {
        return __awaiter(this, void 0, void 0, function* () { });
    }
    static delete() {
        return __awaiter(this, void 0, void 0, function* () { });
    }
}
exports.DetectController = DetectController;
_a = DetectController;
DetectController.create = {
    uploadImage: (imagePath) => __awaiter(void 0, void 0, void 0, function* () {
        // Load the COCO-SSD model
        const model = yield cocoSsd.load();
        // Load the image as a Tensor
        const imageBuffer = fs.readFileSync(imagePath);
        const imageTensor = tf.node.decodeImage(imageBuffer);
        // Run object detection on the image
        const predictions = yield model.detect(imageTensor);
        // Print the predictions
        // console.log('Predictions:');
        // console.log(predictions);
        return predictions;
    }),
    uploadVideo: (videoPath) => __awaiter(void 0, void 0, void 0, function* () {
        const result = [];
        // Load the COCO-SSD model
        const model = yield cocoSsd.load();
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
        command.on('end', () => __awaiter(void 0, void 0, void 0, function* () {
            const frames = fs.readdirSync(framesPath);
            for (let i = 0; i < frames.length; i++) {
                // Load the frame as a Tensor
                const frameBuffer = fs.readFileSync(`${framesPath}/${frames[i]}`);
                const frameTensor = tf.node.decodeImage(frameBuffer);
                // Run object detection on the frame
                const predictions = yield model.detect(frameTensor);
                result.push(predictions);
                // Print the predictions to the console
                console.log(`Frame ${i + 1}:`);
                console.log(predictions);
            }
        }));
        command.run();
        // return result;
    })
};
